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


class MUCapsDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str, tokenizer: LlamaTokenizer, max_words: int):
        print('Load MUCaps dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for audio_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, audio_id))
            self.caption_list.append(one_caption)

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, index):
        filename = self.mm_path_list[index]
        question = ''
        answer = self.caption_list[index]
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


class COCODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str, tokenizer: LlamaTokenizer, max_words: int):
        print('Load COCO dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # keys = random.sample(data.keys(), 10000)
        # data = {k: data[k] for k in keys}
        for video_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, video_id))
            self.caption_list.append(one_caption)

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)])

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, index):
        filename = self.mm_path_list[index]
        question = ''
        answer = self.caption_list[index]
        image = self.transform(Image.open(filename))

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
        return input2, labels, input2_mask, image, "Image", ""


class VideoCapsDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, dataset_type: str, tokenizer: LlamaTokenizer, max_words: int):
        print('Load VideoCaps dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # keys = random.sample(data.keys(), 10000)
        # data = {k: data[k] for k in keys}
        for video_id, one_caption in tqdm(data.items(), total=len(data)):
            self.mm_path_list.append(os.path.join(mm_root_path, video_id))
            self.caption_list.append(one_caption)

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
        self.dataset_type_list = [dataset_type for _ in range(len(self.caption_list))]
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, index):
        filename = self.mm_path_list[index]
        question = ''
        answer = self.caption_list[index]

        container = av.open(filename)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

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
        return input2, labels, input2_mask, video, "Video", ""
