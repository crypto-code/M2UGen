import torch
from torch.utils.data import Dataset
import json
import llama.utils
from transformers import LlamaTokenizer
import copy
import os
import torchaudio
from tqdm.auto import tqdm


class MUCapsDecoderDataset(Dataset):
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
        question = self.caption_list[index]
        answer = self.caption_list[index] + "".join([f"[AUD{i}]" for i in range(8)])
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
        return input2, labels, input2_mask, 0, "Text", self.caption_list[index]