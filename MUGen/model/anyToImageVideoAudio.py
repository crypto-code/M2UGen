import logging
import os.path
from typing import List

import numpy as np
import torch
import gc
from header import *
from torch.cuda.amp import autocast
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import MusicgenForConditionalGeneration
from .layers import *
from .common.utils import *
from .encoders import *
import json

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class MUGen(nn.Module):
    """LoRA for LLaMa model"""

    def __init__(self, **args):
        super(MUGen, self).__init__()
        self.args = args

        self.max_length = args['max_length']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage = args['stage']

        print('args max_length', args['max_length'])
        print('args modality', args['modality'])

        imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt',
                                           self.args['imagebind_version'])

        print(f'Initializing Encoders ...')
        self.mert = MERTEncoder()
        self.mert.to(self.device)
        self.vit = ViTEncoder()
        self.vit.to(self.device)
        self.vivit = ViViTEncoder()
        self.vivit.to(self.device)
        print('Encoders initialized.')

        vicuna_ckpt_path = "./ckpt/pretrained_ckpt/LLaMA-2/7B"  # os.path.join(self.args['pretrained_ckpt_path'], 'vicuna_ckpt', self.args['vicuna_version'])
        print(f'Initializing language decoder from {vicuna_ckpt_path} ...')

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path, device_map="auto", load_in_8bit=True)
        if self.args['freeze_lm']:
            print("Freezing the LLaMa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        else:
            print("Instruct tuning the LLaMa ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            self.llama_model.to(self.device)
        print('Language decoder initialized.')

        # use the new trained tokenizer
        tokenizer_path = vicuna_ckpt_path
        print(f'Initializing tokenizer from {tokenizer_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print(f"Before: {len(self.llama_tokenizer)}")
        # self.llama_tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self._add_audio_token()
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        stops = self.llama_tokenizer('\n###', return_tensors='pt', add_special_tokens=False)
        print(f"After: {len(self.llama_tokenizer)}: {stops}")
        print('Tokenizer initialized.')

        self.input_embeddings = self.llama_model.get_input_embeddings()

        # the alignment module for LLM-TO-AUDIO
        self.ad_ckpt_path = self.args['audio_diffusion']
        self.gen_text_hidden_to_audio = nn.ModuleList([])
        for layer_idx in self.args['text_emb_to_audio_layers']:
            if layer_idx == -1 or layer_idx == self.llama_model.config.num_hidden_layers:
                in_dim = self.llama_model.config.hidden_size

                self.gen_text_hidden_to_audio.append(
                    TextFcLayer(in_dim, 8192,
                                num_input_tokens=self.args['num_gen_audio_tokens'],
                                num_output_tokens=1,
                                mode=self.args['text_fc_to_audio_mode']))
            # self.ad_pipe.text_encoder.config.projection_dim
            elif layer_idx < self.llama_model.config.num_hidden_layers:
                self.gen_text_hidden_to_audio.append(
                    TextFcLayer(self.llama_model.config.hidden_size, 8192,
                                num_input_tokens=self.args['num_gen_audio_tokens'],
                                num_output_tokens=1,
                                mode=self.args['text_fc_to_audio_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llama_model.config.num_hidden_layers} layers.')
        self.gen_text_hidden_to_audio.to(self.device)

        if self.args['freeze_output_proj']:
            for name, param in self.gen_text_hidden_to_audio.named_parameters():
                param.requires_grad = False

    def get_trainable_params(self):
        trainable = {}
        for name, param in self.named_parameters():
            if self.stage == 3 and 'lora' in name:
                trainable[name] = param
            if self.stage in [1, 3] and any(x in name for x in ['mu_mert', 'iu_vit', 'iu_vivit']):
                trainable[name] = param
            if self.stage in [2, 3] and 'gen_text_hidden_to_audio' in name:
                trainable[name] = param
        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True

    def _add_audio_token(self):
        # self.llama_tokenizer.add_tokens({"<Aud>"})  # add special audio token to tokenizer
        # self.llama_tokenizer.add_tokens({"</Aud>"})  # add special audio token to tokenizer

        # Add [AUD] tokens to the vocabulary.
        self.args['gen_audio_token_idx'] = []
        for i in range(self.args['num_gen_audio_tokens']):
            print(f'Adding [AUD{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[AUD{i}]") =',
                  self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens([f'[AUD{i}]'])
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[AUD{i}]") =',
                  self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_audio_token_idx'].append(gen_token_idx[0])

    def encode_video(self, video_paths):
        try:
            inputs_llama = self.vivit(video_paths)  # .to(self.device)   # bsz x 1 x llama_size
        except:
            inputs_llama = torch.zeros((len(video_paths), 1, 4096)).to(self.device)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        inputs_llama = self.mert(audio_paths)  # .to(self.device)   # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs_llama = self.vit(image_paths)  # .to(self.device)  # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2

        batch_size = input_ids.shape[0]

        bos = torch.ones([batch_size, 1], dtype=input_ids.dtype,
                         device=input_ids.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        if self.args['freeze_lm']:
            p_after_embeds = self.llama_model.model.embed_tokens(input_ids).expand(batch_size, -1,
                                                                                   -1)  # bsz x s2 x embed_dim
            bos_embeds = self.llama_model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        else:
            p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1,
                                                                                         -1)  # bsz x s2 x embed_dim
            bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        if img_embeds is not None:
            p_before = '### Human: <Img>'
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                self.device)
            # peft model need deeper call
            if self.args['freeze_lm']:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
            else:
                p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(
                    batch_size, -1, -1)  # bsz x s1 x embed_dim
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1).to(
                self.device)  # bsz x (1+s1+1+s2) x embed_dim

            # create targets
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1],  # 1 (bos) + s1 + 1
                           dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1)
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(
                self.device)  # bsz x (1 + s1 + 1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + 1 + s2)
        else:
            p_before = '### Human: '
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                self.device)
            # peft model need deeper call
            if self.args['freeze_lm']:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
            else:
                p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(
                    batch_size, -1, -1)  # bsz x s1 x embed_dim
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(
                self.device)  # bsz x (1+s1+s2) x embed_dim

            # create targets
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1]],  # 1 (bos) + s1
                           dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1)
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(
                self.device)  # bsz x (1 + s1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + s2)
        return inputs_embeds, targets, attention_mask

    def _train_with_mode(self, texts, img_embeds=None, modality='text', num_gen_tokens='8',
                         text_hidden_fcs=None, gen_token_idx=None, text_emb_layers=None, text_prompt_embeddins=None,
                         loss_scale=1.0, stage=2):
        """
        :param num_gen_tokens: the number of generation tokens
        :param modality: mode can be 'image' / 'video' / 'audio' / 'text'
        :param text_hidden_fcs: alignment module
        :param gen_token_idx: List
        :param text_emb_layers: the layer index of LLM hidden states
        :param text_prompt_embeddins: the textual caption/prompt embeddings
        :param loss_scale: the scale on the mse loss for alignment
        :param stage: the training stage
        :param
        """
        if stage == 2:
            input_ids, target_ids, attention_mask = process_batch_stage_2(self.llama_tokenizer, texts,
                                                                          self.max_length,
                                                                          num_gen_tokens,
                                                                          modality
                                                                          )
        elif stage == 3:
            input_ids, target_ids, attention_mask = process_batch_stage_3(self.llama_tokenizer, texts, self.max_length,
                                                                          self.args['num_gen_audio_tokens'])
        else:
            raise NotImplementedError
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        # print(self.llama_tokenizer.batch_decode(chosen_tokens))
        # print()
        # print(self.llama_tokenizer.batch_decode(labels))
        if modality == 'text':
            del outputs, targets, inputs_embeds, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
            return loss, gen_acc, torch.zeros_like(loss)
        else:
            hidden_states = []
            # text_hidden_fcs = self.gen_text_hidden_fcs

            # based on the targets to obtain the hidden state, targets includes the [BOS] token
            start_pos = (targets == gen_token_idx[0]).nonzero(as_tuple=False)[:, 1].tolist()
            end_pos = (targets == gen_token_idx[-1]).nonzero(as_tuple=False)[:, 1].tolist()
            # print(f'targets : {targets}')
            # print(f'start_pos : {start_pos}')
            # print(f'end_pos : {end_pos}')
            assert 0 < len(start_pos) == len(end_pos) == input_ids.size(0) and len(end_pos) > 0, (start_pos, end_pos)
            for idx, fc_layer in zip(text_emb_layers, text_hidden_fcs):
                hidden_embedding = []
                input_embedding = []
                for b, (s, e) in enumerate(zip(start_pos, end_pos)):
                    assert e - s + 1 == num_gen_tokens, (s, e, num_gen_tokens)
                    hidden_embedding.append(outputs.hidden_states[idx][b, s:e + 1, :])
                    input_embedding.append(self.input_embeddings(targets[b, s:e + 1]))
                hidden_embedding = torch.stack(hidden_embedding, dim=0)
                input_embedding = torch.stack(input_embedding, dim=0)
                hidden_states.append(fc_layer(hidden_embedding, input_embedding))  # (N, seq_len, 2048)
            embeddings = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, 77, 768)

            # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (N, T_I_V_A.txt, 256)

            # Obtain the embeddings produced by the text encoder of a frozen text-to-image generation model
            input_text = [conversation for conversation in texts]

            text_prompt_embeddins = torch.stack(text_prompt_embeddins, dim=0).to(self.device)
            assert len(text_prompt_embeddins.shape) == 2, (text_prompt_embeddins.shape, embeddings.shape)
            text_prompt_embeddins = text_prompt_embeddins.view(text_prompt_embeddins.size(0), 1,
                                                               text_prompt_embeddins.size(1))
            print(embeddings.float().mean(), text_prompt_embeddins.float().mean())
            mse_loss = l2_loss(embeddings, text_prompt_embeddins)
            mse_loss = mse_loss.mean()
            loss += loss_scale * mse_loss
            del outputs, targets, inputs_embeds, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
            return loss, gen_acc, mse_loss

    def _enc_align_training_stage_1(self, inputs):
        """
        In the stage 1: training the encoding-side alignment via image/video/audio caption tasks
        modality: the input modality for each caption task, it could be 'image', 'video' or 'audio'.
        """
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_image(image_paths)
        elif dataset_type == 'VideoToText':
            video_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_video(video_paths)
        elif dataset_type == 'AudioToText':
            audio_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_audio(audio_paths)
        else:
            raise NotImplementedError
        input_ids, target_ids, attention_mask = process_batch_stage_1(self.llama_tokenizer,
                                                                      inputs['output_texts'],
                                                                      self.max_length,
                                                                      self.args['prompt'])
        # print(input_ids)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(mm_embeds, input_ids, target_ids, attention_mask)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc

    def _dec_align_training_stage_2(self, inputs):
        """
        In the stage 2: training the decoding-side alignment via minimize the distance between the
        representation of signal tokens and caption from text encoder within the respective diffusion models.
        modality: the output modality for each caption.
        """
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'TextToAudio':
            loss, gen_acc, mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                            modality='audio',
                                                            num_gen_tokens=self.args['num_gen_audio_tokens'],
                                                            text_hidden_fcs=self.gen_text_hidden_to_audio,
                                                            gen_token_idx=self.args['gen_audio_token_idx'],
                                                            text_emb_layers=self.args['text_emb_to_audio_layers'],
                                                            text_prompt_embeddins=inputs['caption_embs'],
                                                            stage=self.stage)
        else:
            raise NotImplementedError

        return loss, gen_acc, mse_loss

    def _instruction_tuning_stage_3(self, inputs):
        """
        In the stage 3: instruction-following training via the instruction dataset.
        """
        loss = 0
        gen_acc = 0
        mse_loss = []

        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'TextToAudio':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'audio',
                                                            self.args['num_gen_audio_tokens'],
                                                            self.gen_text_hidden_to_audio,
                                                            self.args['gen_audio_token_idx'],
                                                            self.args['text_emb_to_audio_layers'],
                                                            inputs['caption_embs'], stage=self.stage)
        elif dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            img_embeds, _ = self.encode_image(image_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], img_embeds, modality='text',
                                                     stage=self.stage)
        elif dataset_type == 'TextToText':
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], None, modality='text',
                                                     stage=self.stage)
        elif dataset_type == 'AudioToText':
            audio_paths = inputs['mm_paths']
            audio_embeds, _ = self.encode_audio(audio_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], audio_embeds, modality='text',
                                                     stage=self.stage)
        elif dataset_type == "ImageToAudio":
            input_paths = inputs["input_paths"]
            img_embeds, _ = self.encode_image(input_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], img_embeds, 'audio',
                                                     self.args['num_gen_audio_tokens'],
                                                     self.gen_text_hidden_to_audio,
                                                     self.args['gen_audio_token_idx'],
                                                     self.args['text_emb_to_audio_layers'],
                                                     inputs['output_embs'], stage=self.stage)
        elif dataset_type == "VideoToAudio":
            video_paths = inputs['input_paths']
            video_embeds, _ = self.encode_video(video_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], video_embeds, 'audio',
                                                     self.args['num_gen_audio_tokens'],
                                                     self.gen_text_hidden_to_audio,
                                                     self.args['gen_audio_token_idx'],
                                                     self.args['text_emb_to_audio_layers'],
                                                     inputs['output_embs'], stage=self.stage)
        elif dataset_type == "AudioToAudio":
            audio_paths = inputs['input_paths']
            audio_embeds, _ = self.encode_audio(audio_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], audio_embeds, 'audio',
                                                     self.args['num_gen_audio_tokens'],
                                                     self.gen_text_hidden_to_audio,
                                                     self.args['gen_audio_token_idx'],
                                                     self.args['text_emb_to_audio_layers'],
                                                     inputs['output_embs'], stage=self.stage)
        else:
            raise NotImplementedError
        return loss, gen_acc, mse_loss

    def _stage_4_training(self, inputs):
        """
        In the stage 4, we employ the modality-switch dataset to instruction-tune the overall framework
        """
        pass

    def forward(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None
        with autocast():
            if self.stage == 1:
                loss, gen_acc = self._enc_align_training_stage_1(inputs)
            elif self.stage == 2:
                loss, gen_acc, mse_loss = self._dec_align_training_stage_2(inputs)
            elif self.stage == 3:
                loss, gen_acc, mse_loss = self._instruction_tuning_stage_3(inputs)
            else:
                raise NotImplementedError(f"stage {self.stage} is not implemented, now it only support [1, 2, 3]")

        return loss * (1 - gen_acc), gen_acc, mse_loss

    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_paths']:
            image_embeds, _ = self.encode_image(inputs['image_paths'])
            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds

    def _prepare_image_embed(self, text, batch_size):
        pattern = r'Image>(.*?)<\/Image'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                   -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                 -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                         -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                       -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('image path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('image path: ', m)
            _temp_embedding, _ = self.encode_image([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_video_embed(self, text, batch_size):
        pattern = r'Video>(.*?)<\/Video'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                   -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                 -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                         -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                       -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('Video path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('Video path: ', m)
            _temp_embedding, _ = self.encode_video([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_audio_embed(self, text, batch_size):
        pattern = r'Audio>(.*?)<\/Audio'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                   -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                 -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                         -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                       -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('Audio path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('Video path: ', m)
            _temp_embedding, _ = self.encode_audio([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        text = prompt + '\n### Assistant:'
        print("text prompt: ", text)
        batch_size = 1
        input_embeds = []
        split_text = re.split(r' <|> ', text)
        for st in split_text:
            if st.startswith('Image>'):
                input_embeds.append(self._prepare_image_embed(st, batch_size))
            elif st.startswith('Audio>'):
                input_embeds.append(self._prepare_audio_embed(st, batch_size))
            elif st.startswith('Video>'):
                input_embeds.append(self._prepare_video_embed(st, batch_size))
            else:
                text_tokens = self.llama_tokenizer(st, add_special_tokens=False, return_tensors='pt').to(self.device)
                bos = torch.ones([batch_size, 1],
                                 dtype=text_tokens.input_ids.dtype,
                                 device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
                if self.args['freeze_lm']:
                    text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                    bos_embeds = self.llama_model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                else:
                    text_embeds = self.llama_model.model.model.embed_tokens(text_tokens.input_ids).expand(batch_size,
                                                                                                          -1, -1)
                    bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                input_embeds.append(bos_embeds)
                input_embeds.append(text_embeds)
        inputs_embeds = torch.cat(input_embeds, dim=1)  # bsz x (1+s2) x embed_dim
        return inputs_embeds

    def generate_tokens_embeddings(self, inputs, input_embeds, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens and output embeddings that employed to generate images/videos/audios
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
            output_embeddings: output embeddings for synthesizing images
            video_output_embedding: output embeddings for synthesizing video
            audio_output_embedding: output embeddings for synthesizing audio
        """
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            # repeat_pen,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        audio_output_embedding = []
        out = outputs.sequences
        for _hidden_states in outputs.hidden_states[1:]:
            for idx in self.args['text_emb_to_audio_layers']:
                audio_output_embedding.append(_hidden_states[idx])
        audio_output_embedding = torch.cat(audio_output_embedding, dim=1)

        return out, audio_output_embedding

    def generate_audios(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40, audio_length_in_s=5.0):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing audios
        all_gen_idx: the index of [AUD0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = MusicgenForConditionalGeneration.from_pretrained(self.ad_ckpt_path,
                                                                            torch_dtype=torch.float16).to("cuda")
        for gen_idx in all_gen_idx:
            assert generated_ids[0,
                   gen_idx:gen_idx + self.args['num_gen_audio_tokens']].cpu().detach().numpy().tolist() == \
                   self.args[
                       'gen_audio_token_idx'], (
                generated_ids[0, gen_idx:gen_idx + self.args['num_gen_audio_tokens']],
                self.args['gen_audio_token_idx'])
            raw_emb = embeddings[:, gen_idx - 1:gen_idx - 1 + self.args['num_gen_audio_tokens'], :]  # (1, 8, 4096)

            # Produce generation embedding.
            gen_prefix = ''.join([f'[AUD{i}]' for i in range(self.args['num_gen_audio_tokens'])])
            gen_prefx_ids = self.llama_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="pt").input_ids.to(self.device)
            gen_prefix_embs = self.input_embeddings(gen_prefx_ids)  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_to_audio[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)
            # print('gen_emb size:', gen_emb.size())
            bs = gen_emb.shape[0]
            hid_emb_size = gen_emb.shape[2]
            gen_emb = gen_emb.view(bs, hid_emb_size)

            audio_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             audio_length_in_s=audio_length_in_s).audios[0]
            caption = \
                self.llama_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(audio_outputs)
        return return_outputs

    def generate(self, inputs):
        """
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature, Used to modulate logit distribution.
                'modality_embeds': None or torch.tensor,
                'modality_cache': save the image cache,

                'filter_value': Value to assign to tokens that should never be generated,
                'min_word_tokens': Minimum number of words to generate before allowing a [IMG] output.
                'gen_scale_factor': float = 1.0,
                'stops_id': the default value is [[835], [2277, 29937]] the stop token is '###', which has two types of tokenization ways, [835] and [2277, 29937]
                'ENCOUNTERS': the times that the generated sentence will be ended.

                'load_sd': whether use SD for image generation
                'max_num_imgs': Maximum number of images to return in one generation pass.
                'guidance_scale_for_img': the guidance ratio of conditioner, if it is None, the default value will be applied in SD
                'num_inference_steps_for_img': the number of inference step for image generation in the stable diffusion model

                'load_vd': whether use VD for video generation
                'max_num_vids': Maximum number of videos to return in one generation pass.
                'guidance_scale_for_vid': the guidance ratio of conditioner, if it is None, the default value will be applied in VD
                'num_inference_steps_for_vid': the number of inference step for video generation in the stable diffusion model
                'height': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The height in pixels of the generated video.
                'width': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The width in pixels of the generated video.
                'num_frames': (`int`, *optional*, defaults to 16):
                        The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                        amounts to 2 seconds of video.

                'load_ad': whether use AD for audio generation
                'max_num_auds': Maximum number of audios to return in one generation pass.
                'guidance_scale_for_aud': the guidance ratio of conditioner, if it is None, the default value will be applied in AD
                'num_inference_steps_for_aud': the number of inference step for audio generation in the stable diffusion model
                'audio_length_in_s': the seconds for generated audio length
            }
        """
        # init output with image tokens

        input_embeds = self.prepare_generation_embedding(inputs)
        generated_ids, generated_audio_embeddings = self.generate_tokens_embeddings(inputs, input_embeds)

        return_outputs = []

        # Find up to max_num_rest [AUD] tokens, and their corresponding scores.
        print('All ids: ', generated_ids)
        all_gen_aud_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_audio_token_idx'][0]) if x][
                          :inputs['max_num_auds']]
        print('all_gen_aud_idx: ', all_gen_aud_idx)

        if len(all_gen_aud_idx) == 0:
            # No [IMG], [VID], [AUD] tokens.
            caption = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # return_outputs.append(truncate_caption(caption))
            return_outputs.append(caption)
        else:
            if len(all_gen_aud_idx) > 0:
                aud_outputs = self.generate_audios(generated_ids, generated_audio_embeddings, all_gen_aud_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_aud'],
                                                   num_inference_steps=inputs['num_inference_steps_for_aud'],
                                                   audio_length_in_s=inputs['audio_length_in_s'])
                return_outputs.append({'aud': aud_outputs})

        return return_outputs
