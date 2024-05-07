from multiprocessing import Pool
from tqdm.auto import tqdm
import torch
import librosa
from pathlib import Path
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

SAMPLING_RATE = 16000
num_processes = 24  # process_num
model_path = '/home/whatx/.cache/torch/hub/snakers4_silero-vad_master'

def process_music(args):
    music, process_id = args
    cuda_device = f'cuda:{process_id % 2}'  # alternate between cuda:0 and cuda:1
    
    model, utils = torch.hub.load(repo_or_dir=model_path, model='silero_vad', source='local')
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    (get_speech_timestamps, _, _, _, _) = utils

    wav, sr = librosa.load(music, sr=SAMPLING_RATE)
    wav = (wav - np.min(wav)) / (np.max(wav) - np.min(wav))
    speech_timestamps = get_speech_timestamps(torch.Tensor(wav).to(device), model, sampling_rate=SAMPLING_RATE)
    vad = len(speech_timestamps) == 0
    signal_power = np.square(wav).mean()
    noise = np.random.normal(0, 1, len(wav))
    noise_power = np.square(noise).mean()
    snr_value = -10 * np.log10(signal_power / noise_power)
    valid = vad and snr_value >= 10

    return {
        'file': music,
        'snr': snr_value,
        'is_music': vad,
        'valid': valid
    }

if __name__ == "__main__":
    fileset = [str(x) for x in Path("./audioset_full").glob("*.mp3")]
    done_fileset = set()

    # if os.path.exists("./music_filter.jsonl"):
    #     with open('./music_filter.jsonl', 'r') as f:
    #         for line in f:
    #             data = json.loads(line.strip())
    #             done_fileset.add(data['file'])

    remaining_files = [music for music in fileset if music not in done_fileset]
    total_files_to_process = len(remaining_files)
    pbar = tqdm(total=total_files_to_process)

    batch_size = num_processes  # batch_size 
    for i in range(0, len(remaining_files), batch_size):
        batch = remaining_files[i:i + batch_size]
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_music, [(music, idx % num_processes) for idx, music in enumerate(batch)])
        
        # after processing the batch, we can write the results to a file
        # with open('./music_filter.txt', 'a', encoding='utf-8') as f:
        for result in results:
            if not result['valid']:
                os.remove(result['file'])
                # f.write(json.dumps(result, ensure_ascii=False) + '\n')
                # f.write('\n')

        pbar.update(len(batch))

    pbar.close()
