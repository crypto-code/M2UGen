import torch
import librosa
from pathlib import Path
from pprint import pprint
import numpy as np
from tqdm.auto import tqdm
import json
import os
import warnings
warnings.filterwarnings("ignore")

SAMPLING_RATE = 16000

fileset = [str(x) for x in Path("./audioset_full").glob("*.mp3")]

print("Loading Model...")

model_path = '/home/whatx/.cache/torch/hub/snakers4_silero-vad_master'
model, utils = torch.hub.load(repo_or_dir=model_path,
                              model='silero_vad', source='local')# ,
                            #   force_reload=True)
device = torch.device('cuda')
model.to(device)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


class MusicFilter:
    def __init__(self, filename):
        self.wav, self.sr = librosa.load(filename, sr=SAMPLING_RATE)

        # VAD
        wav = (self.wav - np.min(self.wav)) / (np.max(self.wav) - np.min(self.wav))
        speech_timestamps = get_speech_timestamps(torch.Tensor(wav).to(device), model, sampling_rate=SAMPLING_RATE)
        self.vad = len(speech_timestamps) == 0

        # SNR
        signal_power = np.square(self.wav).mean()
        noise = np.random.normal(0, 1, len(self.wav))
        noise_power = np.square(noise).mean()
        self.snr_value = -10 * np.log10(signal_power / noise_power)

    def get_results(self):
        return {"snr": self.snr_value, "is_music": self.vad}

    def __getattr__(self, item):
        if item == "snr":
            return self.snr_value
        elif item == "is_music":
            return self.vad
        elif item == "is_high_quality":
            return self.snr_value >= 10
        elif item == "valid":
            return self.vad and self.snr_value >= 10


music_data = {}
done_fileset = set()
count = 0

if os.path.exists("./music_filter.json"):
    music_data = json.load(open('./music_filter.json', 'r'))
    done_fileset = set(music_data.keys())
    print("Continuing Music Filter...")
else:
    print("Starting Music Filter...")

for music in tqdm(fileset):
    if music in done_fileset:
        if os.path.exists(music):
            if not (music_data[music]["snr"] and music_data[music]["snr"] >=10):
               os.remove(music)
        continue
    mf = MusicFilter(music)
    music_data[music] = mf.get_results()
    if not mf.valid:
        os.remove(music)
    if count % 10 == 0:
        with open('./music_filter.json', 'w') as f:
            json.dump(music_data, f)

with open('./music_filter.json', 'w') as f:
    json.dump(music_data, f)
