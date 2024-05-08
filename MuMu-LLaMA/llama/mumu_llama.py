import json
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama import Transformer, ModelArgs, RMSNorm
from .projector import ProjectionLayer
from util.misc import download
from .utils import sample_top_p
from .musicgen.musicgen import MusicgenForConditionalGeneration
from .audioldm2 import AudioLDM2Pipeline

from transformers import LlamaTokenizer
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from transformers import ViTImageProcessor, ViTModel
from transformers import VivitImageProcessor, VivitModel
from transformers import AutoProcessor

import torchaudio


class MuMu_LLaMA(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, llama_ckpt_dir, llama_tokenizer, model_args, knn=False, knn_dir="./ckpts", stage=1,
                 legacy_bridge=False, load_llama=True, device=None):
        super().__init__()

        self.args = model_args

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. MERT Encoder
        # The model files for MERT can be downloaded here in case of network issues:
        # https://huggingface.co/m-a-p/MERT-v1-330M
        # And set the mert_path argument to directory with the model files
        print(f'Initialize MERT...')
        self.mert_model = AutoModel.from_pretrained(self.args.mert_path, trust_remote_code=True)  # .to(self.device)
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.args.mert_path, trust_remote_code=True)
        self.mu_mert_agg = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        self.mu_mert_rnn = nn.RNN(1024, 1024, batch_first=True)
        self.mu_mert_attention = nn.Linear(1024, 1)
        self.mu_mert_softmax = nn.Softmax(dim=1)
        self.mu_mert_proj = nn.Linear(1024, 4096)

        if legacy_bridge:
            bridge_norm_layer = nn.LayerNorm
            bridge_bias = True
        else:
            bridge_norm_layer = RMSNorm
            bridge_bias = False

        self.feature_scaler = 1

        self.mu_mert_norm_1 = bridge_norm_layer(4096)
        self.mu_mert_f1_1 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.mu_mert_f2_1 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.mu_mert_f3_1 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)

        self.mu_mert_norm_2 = bridge_norm_layer(4096)
        self.mu_mert_f1_2 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.mu_mert_f2_2 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.mu_mert_f3_2 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)

        self.mu_mert_norm_3 = bridge_norm_layer(4096)
        self.mu_mert_f1_3 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.mu_mert_f2_3 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.mu_mert_f3_3 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        print(f'MERT initialized...')

        # 2. ViT Encoder
        # The model files for ViT can be downloaded here in case of network issues:
        # https://huggingface.co/google/vit-base-patch16-224-in21k
        # And set the vit_path argument to directory with the model files
        print(f'Initialize ViT...')
        self.vit_model = ViTModel.from_pretrained(self.args.vit_path)  # .to(self.device)
        self.vit_model.eval()
        self.vit_processor = ViTImageProcessor.from_pretrained(self.args.vit_path, do_rescale=False)
        self.iu_vit_agg = nn.Conv1d(in_channels=197, out_channels=1, kernel_size=1)
        self.iu_vit_proj = nn.Linear(768, 4096)

        self.iu_vit_norm_1 = bridge_norm_layer(4096)
        self.iu_vit_f1_1 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.iu_vit_f2_1 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.iu_vit_f3_1 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)

        self.iu_vit_norm_2 = bridge_norm_layer(4096)
        self.iu_vit_f1_2 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.iu_vit_f2_2 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.iu_vit_f3_2 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)

        self.iu_vit_norm_3 = bridge_norm_layer(4096)
        self.iu_vit_f1_3 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.iu_vit_f2_3 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.iu_vit_f3_3 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        print(f'ViT initialized...')

        # 3. ViViT Encoder
        # The model files for ViViT can be downloaded here in case of network issues:
        # https://huggingface.co/google/vivit-b-16x2-kinetics400
        # And set the vivit_path argument to directory with the model files
        print(f'Initialize ViViT...')
        self.vivit_model = VivitModel.from_pretrained(self.args.vivit_path)  # .to(self.device)
        self.vivit_model.eval()
        self.vivit_processor = VivitImageProcessor.from_pretrained(self.args.vivit_path)
        self.iu_vivit_agg = nn.Conv1d(in_channels=3137, out_channels=1, kernel_size=1)
        self.iu_vivit_rnn = nn.RNN(1024, 1024, batch_first=True)
        self.iu_vivit_attention = nn.Linear(1024, 1)
        self.iu_vivit_softmax = nn.Softmax(dim=1)
        self.iu_vivit_proj = nn.Linear(768, 4096)

        self.iu_vivit_norm_1 = bridge_norm_layer(4096)
        self.iu_vivit_f1_1 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.iu_vivit_f2_1 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.iu_vivit_f3_1 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)

        self.iu_vivit_norm_2 = bridge_norm_layer(4096)
        self.iu_vivit_f1_2 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.iu_vivit_f2_2 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.iu_vivit_f3_2 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)

        self.iu_vivit_norm_3 = bridge_norm_layer(4096)
        self.iu_vivit_f1_3 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        self.iu_vivit_f2_3 = nn.Linear(4096 * self.feature_scaler, 4096, bias=bridge_bias)
        self.iu_vivit_f3_3 = nn.Linear(4096, 4096 * self.feature_scaler, bias=bridge_bias)
        print(f'ViViT initialized...')

        # 4. llama
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        bias_lora = True

        if self.args.music_decoder.lower() == "audioldm2":
            self.model_args: ModelArgs = ModelArgs(
                max_seq_len=1024, max_batch_size=1, w_bias=bias_lora, w_lora=bias_lora,
                num_output_tokens=1, output_dim_tokens=137216,
                **params)  # max_batch_size only affects inference
        else:
            self.model_args: ModelArgs = ModelArgs(
                max_seq_len=1024, max_batch_size=1, w_bias=bias_lora, w_lora=bias_lora,
                num_output_tokens=128, output_dim_tokens=768,
                **params)  # max_batch_size only affects inference
        print(f"model args: {self.model_args}")

        # 5. tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            llama_tokenizer)  # Tokenizer(model_path=llama_tokenizer, num_aud_tokens=self.model_args.num_gen_audio_tokens)
        self._add_audio_token()
        self.model_args.vocab_size = len(self.tokenizer)

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(self.model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        if load_llama:
            print(f"Loading LLaMA Checkpoint...")
            ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))

            """
            Adapted from https://github.com/cedrickchee/llama/blob/main/chattyllama/combined/inference.py
            """
            key_to_dim = {
                "w1": 0,
                "w2": -1,
                "w3": 0,
                "wo": -1,
                "wq": 0,
                "wk": 0,
                "wv": 0,
                "output": 0,
                "tok_embeddings": 2,
                "ffn_norm": None,
                "attention_norm": None,
                "norm": None,
                "rope": None,
            }
            for i, ckpt in enumerate(ckpts):
                checkpoint = torch.load(ckpt, map_location="cpu")
                for parameter_name, parameter in self.llama.named_parameters():
                    short_name = parameter_name.split(".")[-2]
                    if "gate" in parameter_name or "lora" in parameter_name or "bias" in parameter_name:
                        continue
                    if key_to_dim[short_name] is None and i == 0:
                        parameter.data = checkpoint[parameter_name]
                    elif key_to_dim[short_name] == 0:
                        size = checkpoint[parameter_name].size(0)
                        parameter.data[size * i: size * (i + 1), :] = checkpoint[
                            parameter_name
                        ]
                    elif key_to_dim[short_name] == -1:
                        size = checkpoint[parameter_name].size(-1)
                        parameter.data[:, size * i: size * (i + 1)] = checkpoint[
                            parameter_name
                        ]
                    elif key_to_dim[short_name] == 2:
                        size = checkpoint[parameter_name].size(-1)
                        parameter.data[:-self.model_args.num_gen_audio_tokens, size * i: size * (i + 1)] = checkpoint[
                            parameter_name
                        ]
                        parameter.data[-self.model_args.num_gen_audio_tokens:, :] = 1
                del checkpoint
            print(f"LLaMA Checkpoint Loaded")

        # 5. projector
        self.output_projector = ProjectionLayer(4096, self.model_args.output_dim_tokens,
                                                num_input_tokens=self.model_args.num_gen_audio_tokens,
                                                num_output_tokens=self.model_args.num_output_tokens)

        # 6. Generator
        if self.args.music_decoder.lower() == "audioldm2":
            # The model files for AudioLDM2 can be downloaded here in case of network issues:
            # https://huggingface.co/cvssp/audioldm2-music
            # And set the music_decoder_path argument to directory with the model files
            print(f'Initialize AudioLDM2...')
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.generation_model = AudioLDM2Pipeline.from_pretrained(self.args.music_decoder_path, torch_dtype=dtype)
            self.generation_model.to("cuda")
            print(f'AudioLDM2 initialized...')
        else:
            # The model files for MusicGen can be downloaded here in case of network issues:
            # https://huggingface.co/facebook/musicgen-medium
            # And set the music_decoder_path argument to directory with the model files
            print(f'Initialize MusicGen...')
            self.generation_processor = AutoProcessor.from_pretrained(self.args.music_decoder_path)
            self.generation_model = MusicgenForConditionalGeneration.from_pretrained(self.args.music_decoder_path)
            self.generation_model.eval()
            print(f'MusicGen initialized...')
        self.music_decoder = self.args.music_decoder.lower()

        # 4. prefix
        self.query_layer = 6
        self.query_len = 1
        self.prefix_query = nn.Embedding(self.query_layer * 3 * self.query_len, self.model_args.dim)

        # 5. knn
        self.knn = knn
        if knn:
            import faiss
            self.index = faiss.read_index(download("https://huggingface.co/csuhan/knn/resolve/main/knn.index", knn_dir))
            print(self.index)

        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.l2_loss = torch.nn.MSELoss()
        self.stage = stage
        self.set_default_trainability(self.stage)

    def get_trainable_params(self, stage=1):
        trainable = {}
        if stage == 1:
            for name, para in self.named_parameters():
                if "llama." in name:
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                if "mu_mert_" in name:
                    trainable[name] = para
                if "iu_vivit_" in name:
                    trainable[name] = para
                if "iu_vit_" in name:
                    trainable[name] = para
                if "prefix_query" in name:
                    trainable[name] = para
                if "tok_embeddings" in name:
                    trainable[name] = para
        elif stage == 2:
            for name, para in self.named_parameters():
                if "llama." in name:
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                if "output_projector" in name:
                    trainable[name] = para
                if "prefix_query" in name:
                    trainable[name] = para
                if "tok_embeddings" in name:
                    trainable[name] = para
        elif stage == 3:
            for name, para in self.named_parameters():
                if "llama." in name:
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                if "prefix_query" in name:
                    trainable[name] = para
                if "tok_embeddings" in name:
                    trainable[name] = para
        return trainable

    def set_default_trainability(self, stage=1):
        for key, value in self.named_parameters():
            value.requires_grad = False
        trainable_params = self.get_trainable_params(stage)
        print(f"Trainable Params: {trainable_params.keys()}")
        for key, value in trainable_params.items():
            value.data = value.data.float()
            value.requires_grad = True

    def _add_audio_token(self):
        self.audio_tokens = []
        for i in range(self.model_args.num_gen_audio_tokens):
            print(f'Adding [AUD{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[AUD{i}]") =',
                  self.tokenizer(f'[AUD{i}]', add_special_tokens=False))
            num_added_tokens = self.tokenizer.add_tokens([f'[AUD{i}]'])
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[AUD{i}]") =',
                  self.tokenizer(f'[AUD{i}]', add_special_tokens=False), ' Number of tokens: ', len(self.tokenizer))
            gen_token_idx = self.tokenizer(f'[AUD{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.audio_tokens.append(gen_token_idx[0])

    def load_audio(self, audio_path, target_sr=16000):
        y, sr = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sr, target_sr, dtype=y.dtype)
        audio = resampler(y)
        return audio, target_sr

    def encode_audio(self, x):
        xs = []
        for sub_x in x:
            all_inputs = [self.mert_processor(sub_x[ix * self.mert_processor.sampling_rate:min(
                (ix + 60) * self.mert_processor.sampling_rate, len(sub_x))],
                                              sampling_rate=self.mert_processor.sampling_rate,
                                              return_tensors="pt").to(self.device) for ix in
                          range(0, len(sub_x) // (self.mert_processor.sampling_rate * 60) + 1, 60)]
            aggoutputs = []
            for inputs in all_inputs:
                with torch.no_grad():
                    outputs = self.mert_model(**inputs, output_hidden_states=True)
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                sub_x = torch.swapaxes(all_layer_hidden_states, 0, 1)
                aggoutputs.append(sub_x)
            aggoutputs = torch.cat(aggoutputs)
            sub_x = self.mu_mert_agg(aggoutputs).squeeze()
            xs.append(sub_x)
        x = torch.stack(xs, dim=0)
        return x

    def encode_image(self, x):
        xs = []
        for sub_x in x:
            inputs = self.vit_processor(images=sub_x, return_tensors="pt").to(self.vit_model.device)
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            sub_x = self.iu_vit_agg(last_hidden_states.to(self.device)).squeeze()
            xs.append(sub_x)
        return torch.stack(xs, dim=0)

    def encode_video(self, x):
        xs = []
        for sub_x in x:
            inputs = self.vivit_processor(list(sub_x), padding=True, return_tensors="pt").to(self.vivit_model.device)
            with torch.no_grad():
                outputs = self.vivit_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.output_hidden_states
            sub_x = self.iu_vivit_agg(hidden_states.to(self.device)).squeeze()
            xs.append(sub_x)
        return torch.stack(xs, dim=0)

    def forward_audio(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_audio(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        audio_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = audio_feats.device

        audio_feats, _ = self.mu_mert_rnn(audio_feats)

        attention_weights = self.mu_mert_attention(audio_feats).squeeze(-1)
        attention_scores = self.mu_mert_softmax(attention_weights)
        
        audio_feats = torch.matmul(attention_scores.unsqueeze(1), audio_feats).squeeze(1)

        if self.knn:
            audio_feats_ori = audio_feats
            sims, indices = self.index.search(audio_feats.cpu(), int(cache_size))
            B = sims.shape[0]
            prototypes = [self.index.reconstruct(x) for x in indices.reshape(-1, ).tolist()]
            prototypes = np.vstack(prototypes).reshape(B, int(cache_size), -1)  # [N, top_k, 1024]
            sims = torch.tensor(sims, device=device)
            prototypes = torch.tensor(prototypes, device=device)

            sims = (sims * cache_t).softmax(dim=-1)
            audio_feats = sims @ prototypes
            audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)

            audio_feats = (1 - cache_weight) * audio_feats_ori + cache_weight * audio_feats
            audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)

        audio_feats = audio_feats.unsqueeze(1)  # B, 1, D
        audio_feats = self.mu_mert_proj(audio_feats)
        audio_feats_norm = self.mu_mert_norm_1(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_1(
            F.silu(self.mu_mert_f1_1(audio_feats_norm)) * self.mu_mert_f3_1(audio_feats_norm))

        audio_feats_norm = self.mu_mert_norm_2(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_2(
            F.silu(self.mu_mert_f1_2(audio_feats_norm)) * self.mu_mert_f3_2(audio_feats_norm))

        audio_feats_norm = self.mu_mert_norm_3(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_3(
            F.silu(self.mu_mert_f1_3(audio_feats_norm)) * self.mu_mert_f3_3(audio_feats_norm))
        return audio_feats

    def forward_image(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_image(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        image_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = image_feats.device

        if self.knn:
            image_feats_ori = image_feats
            sims, indices = self.index.search(image_feats.cpu(), int(cache_size))
            B = sims.shape[0]
            prototypes = [self.index.reconstruct(x) for x in indices.reshape(-1, ).tolist()]
            prototypes = np.vstack(prototypes).reshape(B, int(cache_size), -1)  # [N, top_k, 1024]
            sims = torch.tensor(sims, device=device)
            prototypes = torch.tensor(prototypes, device=device)

            sims = (sims * cache_t).softmax(dim=-1)
            image_feats = sims @ prototypes
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            image_feats = (1 - cache_weight) * image_feats_ori + cache_weight * image_feats
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        image_feats = image_feats.unsqueeze(1)  # B, 1, D
        image_feats = self.iu_vit_proj(image_feats)
        image_feats_norm = self.iu_vit_norm_1(image_feats)
        image_feats = image_feats + self.iu_vit_f2_1(
            F.silu(self.iu_vit_f1_1(image_feats_norm)) * self.iu_vit_f3_1(image_feats_norm))

        image_feats_norm = self.iu_vit_norm_2(image_feats)
        image_feats = image_feats + self.iu_vit_f2_2(
            F.silu(self.iu_vit_f1_2(image_feats_norm)) * self.iu_vit_f3_2(image_feats_norm))

        image_feats_norm = self.iu_vit_norm_3(image_feats)
        image_feats = image_feats + self.iu_vit_f2_3(
            F.silu(self.iu_vit_f1_3(image_feats_norm)) * self.iu_vit_f3_3(image_feats_norm))
        return image_feats

    def forward_video(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_video(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        video_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = video_feats.device

        video_feats, _ = self.iu_vivit_rnn(video_feats)

        attention_weights = self.iu_vivit_attention(video_feats).squeeze(-1)
        attention_scores = self.iu_vivit_softmax(attention_weights)
        
        video_feats = torch.matmul(attention_scores.unsqueeze(1), video_feats).squeeze(1)

        if self.knn:
            video_feats_ori = video_feats
            sims, indices = self.index.search(video_feats.cpu(), int(cache_size))
            B = sims.shape[0]
            prototypes = [self.index.reconstruct(x) for x in indices.reshape(-1, ).tolist()]
            prototypes = np.vstack(prototypes).reshape(B, int(cache_size), -1)  # [N, top_k, 1024]
            sims = torch.tensor(sims, device=device)
            prototypes = torch.tensor(prototypes, device=device)

            sims = (sims * cache_t).softmax(dim=-1)
            video_feats = sims @ prototypes
            video_feats = video_feats / video_feats.norm(dim=-1, keepdim=True)

            video_feats = (1 - cache_weight) * video_feats_ori + cache_weight * video_feats
            video_feats = video_feats / video_feats.norm(dim=-1, keepdim=True)

        video_feats = video_feats.unsqueeze(1)  # B, 1, D
        video_feats = self.iu_vivit_proj(video_feats)
        video_feats_norm = self.iu_vivit_norm_1(video_feats)
        video_feats = video_feats + self.iu_vivit_f2_1(
            F.silu(self.iu_vivit_f1_1(video_feats_norm)) * self.iu_vivit_f3_1(video_feats_norm))

        video_feats_norm = self.iu_vivit_norm_2(video_feats)
        video_feats = video_feats + self.iu_vivit_f2_2(
            F.silu(self.iu_vivit_f1_2(video_feats_norm)) * self.iu_vivit_f3_2(video_feats_norm))

        video_feats_norm = self.iu_vivit_norm_3(video_feats)
        video_feats = video_feats + self.iu_vivit_f2_3(
            F.silu(self.iu_vivit_f1_3(video_feats_norm)) * self.iu_vivit_f3_3(video_feats_norm))
        return video_feats

    @torch.inference_mode()
    def forward_inference(self, tokens, start_pos: int, audio_feats=None, image_feats=None, video_feats=None):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos:start_pos + seqlen]

        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        music_output_embedding = []
        for layer in self.llama.layers[:-3 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)
            music_output_embedding.append(h)

        prefix_query = self.prefix_query.weight.reshape(
            self.query_layer * 3, 1, 4096).unsqueeze(1)

        prefix_index = 0
        if audio_feats is not None:
            for layer in self.llama.layers[-3 * self.query_layer:-2 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, audio_feats + prefix_query[prefix_index])
                music_output_embedding.append(h)
                prefix_index = prefix_index + 1
        else:
            for layer in self.llama.layers[-3 * self.query_layer:-2 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, prefix_query[prefix_index])
                music_output_embedding.append(h)
                prefix_index = prefix_index + 1

        if image_feats is not None:
            for layer in self.llama.layers[-2 * self.query_layer:-1 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, image_feats + prefix_query[prefix_index])
                music_output_embedding.append(h)
                prefix_index = prefix_index + 1
        else:
            for layer in self.llama.layers[-2 * self.query_layer:-1 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, prefix_query[prefix_index])
                music_output_embedding.append(h)
                prefix_index = prefix_index + 1

        if video_feats is not None:
            for layer in self.llama.layers[-1 * self.query_layer:]:
                h = layer(h, 0, freqs_cis, mask, video_feats + prefix_query[prefix_index])
                music_output_embedding.append(h)
                prefix_index = prefix_index + 1
        else:
            for layer in self.llama.layers[-1 * self.query_layer:]:
                h = layer(h, 0, freqs_cis, mask, prefix_query[prefix_index])
                music_output_embedding.append(h)
                prefix_index = prefix_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float(), torch.cat(music_output_embedding[-1:], dim=1)

    def forward(self, tokens, labels, audios=None, imgs=None, videos=None, music_caption=None):
        audio_feats, video_feats, image_feats = None, None, None
        if audios is not None:
            audio_feats = self.forward_audio({'Audio': [audios, 1]})
        if videos is not None:
            video_feats = self.forward_video({'Video': [videos, 1]})
        if imgs is not None:
            image_feats = self.forward_image({'Image': [imgs, 1]})
        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens.to(self.device))
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-3 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)
        prefix_query = self.prefix_query.weight.reshape(
            self.query_layer * 3, 1, 4096).unsqueeze(1)

        prefix_index = 0
        if audio_feats is not None:
            for layer in self.llama.layers[-3 * self.query_layer:-2 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, audio_feats + prefix_query[prefix_index])
                prefix_index = prefix_index + 1
        else:
            for layer in self.llama.layers[-3 * self.query_layer:-2 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, prefix_query[prefix_index])
                prefix_index = prefix_index + 1

        if image_feats is not None:
            for layer in self.llama.layers[-2 * self.query_layer:-1 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, image_feats + prefix_query[prefix_index])
                prefix_index = prefix_index + 1
        else:
            for layer in self.llama.layers[-2 * self.query_layer:-1 * self.query_layer]:
                h = layer(h, 0, freqs_cis, mask, prefix_query[prefix_index])
                prefix_index = prefix_index + 1

        if video_feats is not None:
            for layer in self.llama.layers[-1 * self.query_layer:]:
                h = layer(h, 0, freqs_cis, mask, video_feats + prefix_query[prefix_index])
                prefix_index = prefix_index + 1
        else:
            for layer in self.llama.layers[-1 * self.query_layer:]:
                h = layer(h, 0, freqs_cis, mask, prefix_query[prefix_index])
                prefix_index = prefix_index + 1

        final_hidden = h
        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000 + self.model_args.num_gen_audio_tokens, self.llama.vocab_size
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten().to(self.device))

        if music_caption is not None and any([mc != '' for mc in music_caption]):
            if not all([i in output for i in range(32000, 32008)]):
                c_loss += 100
            if self.music_decoder == "audioldm2":
                prompt_embeds, generated_prompt_embeds = self.generation_model(prompt=list(music_caption),
                                                                               guidance_scale=1,
                                                                               return_prompts_only=True)
                prompt_embeds = prompt_embeds.reshape(prompt_embeds.shape[0], -1)
                generated_prompt_embeds = generated_prompt_embeds.reshape(generated_prompt_embeds.shape[0], -1)
                out_embed = torch.cat([prompt_embeds, generated_prompt_embeds], dim=1)
                out_embed = 10 * out_embed.view(out_embed.size(0), 1, out_embed.size(1)).to(self.device)
            else:
                gen_inputs = self.generation_processor(text=music_caption, padding='max_length',
                                                       max_length=128, truncation=True, return_tensors="pt").to(
                    self.device)
                out_embed = 10 * self.generation_model.generate(**gen_inputs, guidance_scale=1, encoder_only=True)
                del gen_inputs
            start_pos = (labels == self.audio_tokens[0]).nonzero(as_tuple=False)[:, 1].tolist()
            assert len(start_pos) != 0, (self.tokenizer.batch_decode(labels), music_caption)
            hidden_states = []
            hidden_embedding = []
            input_embedding = []
            for b, s in enumerate(start_pos):
                hidden_embedding.append(final_hidden[b, s:s + self.model_args.num_gen_audio_tokens, :])
                input_embedding.append(
                    self.llama.tok_embeddings(labels[b, s:s + self.model_args.num_gen_audio_tokens].to(self.device)))
            hidden_embedding = torch.stack(hidden_embedding, dim=0).to(self.device)
            input_embedding = torch.stack(input_embedding, dim=0).to(self.device)
            hidden_states.append(self.output_projector(hidden_embedding, input_embedding))
            embeddings = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            mse_loss = self.l2_loss(embeddings, out_embed)
            del hidden_states, input_embedding, hidden_embedding, out_embed, embeddings
            # c_loss += mse_loss
        else:
            if any([i in output for i in range(32000, 32008)]):
                c_loss += 100
            mse_loss = torch.tensor(0.0)
        return c_loss, mse_loss

    @torch.inference_mode()
    def generate_music(self, embeddings, audio_length_in_s, music_caption):
        gen_prefix = ''.join([f'[AUD{i}]' for i in range(len(self.audio_tokens))])
        gen_prefx_ids = self.tokenizer(gen_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(
            self.device)
        gen_prefix_embs = self.llama.tok_embeddings(gen_prefx_ids)
        if self.music_decoder == "audioldm2":
            gen_emb = self.output_projector(embeddings.float().to("cuda"), gen_prefix_embs).squeeze(dim=0) / 10
            prompt_embeds, generated_prompt_embeds = gen_emb[:, :128 * 1024], gen_emb[:, 128 * 1024:]
            prompt_embeds = prompt_embeds.reshape(prompt_embeds.shape[0], 128, 1024)
            generated_prompt_embeds = generated_prompt_embeds.reshape(generated_prompt_embeds.shape[0], 8, 768)
            print("Generating Music...")
            audio_outputs = self.generation_model(music_caption,
                                                  num_inference_steps=200,
                                                  num_waveforms_per_prompt=3,
                                                  negative_prompt='Low quality.',
                                                  audio_length_in_s=audio_length_in_s).audios
            return audio_outputs
        else:
            print("Generating Music...")
            gen_emb = 0.1 * self.output_projector(embeddings.float().to("cuda"), gen_prefix_embs) / 10
            gen_inputs = self.generation_processor(text=music_caption, padding='max_length',
                                                   max_length=128, truncation=True, return_tensors="pt").to(
                self.device)
            audio_outputs = self.generation_model.generate(**gen_inputs, guidance_scale=3.5,
                                                           max_new_tokens=int(256 / 5 * audio_length_in_s))
            return audio_outputs[0][0].cpu().detach().numpy()

    @torch.inference_mode()
    def generate(
            self,
            prompts,
            audios=None,
            imgs=None,
            videos=None,
            max_gen_len: int = 256,
            temperature: float = 0.1,
            top_p: float = 0.75,
            cache_size=10,
            cache_t=20,
            cache_weight=0.5,
            audio_length_in_s=10
    ):
        bsz = len(prompts)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        with torch.cuda.amp.autocast():
            if audios is not None:
                audio_feats = self.forward_audio({'Audio': [[audios], 1]}, cache_size, cache_t, cache_weight)
            else:
                audio_feats = None
            if videos is not None:
                video_feats = self.forward_video({'Video': [[videos], 1]}, cache_size, cache_t, cache_weight)
            else:
                video_feats = None
            if imgs is not None:
                image_feats = self.forward_image({'Image': [[imgs], 1]}, cache_size, cache_t, cache_weight)
            else:
                image_feats = None

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer(x).input_ids[:, 1:] for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != 0
        start_pos = min_prompt_size
        prev_pos = 0
        music_output_embeddings = []
        start_gather = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits, music_output_embedding = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos,
                                                                        audio_feats, image_feats, video_feats)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if next_token[0] == self.audio_tokens[start_gather]:
                if start_gather == 0:
                    music_output_embeddings = []
                music_output_embeddings.append(music_output_embedding[:, -1:, :])
                start_gather += 1
                if start_gather >= len(self.audio_tokens):
                    start_gather = 0
                    break
            # trick: early stop if bsz==1
            if bsz == 1 and self.tokenizer.decode(tokens[0, cur_pos - 2:cur_pos + 1]) == "\n###":
                break
            # prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(13)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        if len(music_output_embeddings) == len(self.audio_tokens):
            music_output_embeddings = torch.cat(music_output_embeddings, dim=1)
            return [decoded[0], {'aud': [self.generate_music(music_output_embeddings, audio_length_in_s, decoded[0])]}]

        return [decoded[0]]


def load(model_path, llama_dir, mert_path="m-a-p/MERT-v1-330M", device="cuda" if torch.cuda.is_available() else "cpu",
         knn=False, knn_dir="./ckpts", llama_type="7B", stage=3):
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = llama_dir

    # load MuMu_LLaMA weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    adapter_ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = adapter_ckpt.get('config', {})

    # The model files for MERT can be downloaded here in case of network issues:
    # https://huggingface.co/m-a-p/MERT-v1-330M
    # And set the MERT argument to directory with the model files
    model = MuMu_LLaMA(
        llama_ckpt_dir, llama_tokenzier_path, mert_path, knn=knn, knn_dir=knn_dir, stage=stage)

    load_result = model.load_state_dict(adapter_ckpt['model'], strict=False)
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device)
