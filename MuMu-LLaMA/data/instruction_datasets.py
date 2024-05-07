import torch
from torch.utils.data import Dataset
import json
import llama.utils
from transformers import LlamaTokenizer
import copy
import os
import numpy as np
import torchaudio
from PIL import Image
import av
from tqdm.auto import tqdm
from torchvision import transforms


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len > seg_len:
        converted_len = 0
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class AnyToMusicInstructionDataset(Dataset):
    """
    T + X - T + X instruction Dataset
    """

    def __init__(self, data_path: str, input_root_path: str, output_root_path: str, dataset_type: str,
                 tokenizer: LlamaTokenizer, max_words: int):
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.instruction_list = []
        self.input_path_list = []
        self.output_path_list = []
        self.caption_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)])
        for instance in tqdm(res, total=len(res)):
            try:
                filename = os.path.join(input_root_path, instance['input_file'])
                if dataset_type == "VideoToAudio":
                    container = av.open(filename)
                    indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                    feats = read_video_pyav(container=container, indices=indices)
                if 'conversation' in instance:
                    self.instruction_list.append(instance['conversation'])
                else:
                    self.instruction_list.append(instance['conversations'])
                self.input_path_list.append(os.path.join(input_root_path, instance['input_file']))
                self.output_path_list.append(os.path.join(output_root_path, instance['output_file']))
                if 'conversation' in instance:
                    self.caption_list.append(instance['conversation'][-1]['caption'])
                else:
                    self.caption_list.append(instance['conversations'][-1]['caption'])
            except:
                continue
            
        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]
        print(f"Total Datapoints: {len(self.instruction_list)}")

    def __len__(self):  # number of instances
        return len(self.instruction_list)

    def __getitem__(self, index):
        # with open(os.path.join(self.embed_path, str(os.path.basename(self.output_path_list[i])) + '.npy'), 'rb') as f:
        #     output_embs = torch.from_numpy(np.load(f, allow_pickle=True))
        filename = self.input_path_list[index]
        if self.dataset_type_list[index] == "ImageToAudio":
            modality = "Image"
            feats = self.transform(Image.open(filename))
        elif self.dataset_type_list[index] == "AudioToAudio":
            modality = "Audio"
            sample_rate = 24000
            waveform, sr = torchaudio.load(filename)
            if sample_rate != sr:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
            feats = torch.mean(waveform, 0)
        elif self.dataset_type_list[index] == "VideoToAudio":
            modality = "Video"
            container = av.open(filename)
            indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            feats = read_video_pyav(container=container, indices=indices)

        music = self.caption_list[index]

        question = self.instruction_list[index][0]['value']
        answer = self.instruction_list[index][-1]['value'] + " " + "".join([f"[AUD{i}]" for i in range(8)])

        format_instruction = question
        input1 = llama.utils.format_prompt(format_instruction)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, feats, modality, music


class MusicQADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str, tokenizer: LlamaTokenizer, max_words: int):
        print('Load MusicQA dataset ...')
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.mm_root_path = mm_root_path
        self.instruction_list = []
        self.mm_path_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for instance in tqdm(res, total=len(res)):
            if 'conversation' in instance:
                self.instruction_list.append(instance['conversation'])
            else:
                self.instruction_list.append(instance['conversations'])
            self.mm_path_list.append(os.path.join(mm_root_path, instance['audio_name']))
        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]

    def __len__(self):
        return len(self.instruction_list)

    def __getitem__(self, index):
        filename = self.mm_path_list[index]
        question = self.instruction_list[index][0]['value']
        answer = self.instruction_list[index][-1]['value']
        sample_rate = 24000
        waveform, sr = torchaudio.load(filename)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        audio = torch.mean(waveform, 0)

        format_instruction = question
        input1 = llama.utils.format_prompt(format_instruction)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, audio, "Audio", ""


class AlpacaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, dataset_type: str, tokenizer: LlamaTokenizer, max_words: int):
        print('Load Alpaca dataset ...')
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.instruction_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for instance in tqdm(res, total=len(res)):
            self.instruction_list.append(instance['conversation'])
        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]

    def __len__(self):
        return len(self.instruction_list)

    def __getitem__(self, index):
        question = self.instruction_list[index][0]['value']
        answer = self.instruction_list[index][-1]['value']

        format_instruction = question
        input1 = llama.utils.format_prompt(format_instruction)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, 0, "Text", ""
