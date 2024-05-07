from .enc_datasets import *
from .dec_datasets import *
from .instruction_datasets import *
from torch.utils.data import Dataset, ConcatDataset


class FinetuneDataset(Dataset):
    def __init__(self, max_words=30, tokenizer=None, stage=1):
        dataset_list = []

        if stage == 1:
            # Encoder Datasets
            mucaps = MUCapsDataset("./Datasets/MUCaps/MUCapsCaptions.json",
                                   "./Datasets/MUCaps/audios/", "AudioToText", tokenizer, max_words)
            coco = COCODataset("./Datasets/COCO/COCOCaptions.json",
                               "./Datasets/COCO/train2014/", "ImageToText", tokenizer, max_words)
            videocaps = VideoCapsDataset("./Datasets/MUVideo/MUVideoCaptions.json",
                                         "./Datasets/MUVideo/audioset_video/", "VideoToText",
                                         tokenizer, max_words)
            dataset_list.append(mucaps)
            dataset_list.append(coco)
            dataset_list.append(videocaps)

        if stage == 2:
            # Decoder Dataset
            mucaps_decoder = MUCapsDecoderDataset("./Datasets/MUCaps/MUCapsCaptions.json",
                                                  "./Datasets/MUCaps/audios/", "TextToAudio",
                                                  tokenizer, max_words)
            dataset_list.append(mucaps_decoder)

        if stage == 3:
            # QA Dataset
            musicqa2 = MusicQADataset("./Datasets/MusicQAv2.0/MusicQAv2.json",
                                     "./Datasets/MusicQAv2.0", "AudioToText", tokenizer,
                                     max_words)
            musicqa_gpt = MusicQADataset("./Datasets/MusicQAv2.0/MusicQA_chatgpt.json",
                                     "./Datasets/MusicQAv2.0", "AudioToText", tokenizer,
                                     max_words)

            # Text Instruction
            alpaca = AlpacaDataset("./Datasets/Alpaca/alpaca_data.json", "TextToText", tokenizer,
                                   max_words)

            # Generation Instruction Datasets
            muimage = AnyToMusicInstructionDataset("./MUDataset/MUImage_Instructions.json",
                                                   "./MUDataset",
                                                   "./MUDataset",
                                                   "ImageToAudio", tokenizer, max_words)
            muvideo = AnyToMusicInstructionDataset("./MUDataset/MUVideo_Instructions.json",
                                                   "./MUDataset",
                                                   "./MUDataset",
                                                   "VideoToAudio", tokenizer, max_words)
            muedit = AnyToMusicInstructionDataset("./Datasets/MUEdit/MUEditInstructions.json",
                                                  "./Datasets/MUEdit/audioset",
                                                  "./Datasets/MUEdit/audioset",
                                                  "AudioToAudio", tokenizer, max_words)
            dataset_list.append(musicqa2)
            dataset_list.append(musicqa_gpt)
            dataset_list.append(alpaca)
            dataset_list.append(muimage)
            dataset_list.append(muvideo)
            dataset_list.append(muedit)
        self.datasets = ConcatDataset(dataset_list)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]