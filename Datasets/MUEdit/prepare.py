import json, os, warnings, argparse, math
from tqdm import tqdm
import torch, librosa
from pathlib import Path
import numpy as np
from collections import Counter
warnings.filterwarnings("ignore")

SAMPLING_RATE = 16000

# load the json file
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
    
# get the origin audio list
def get_audio_list(dir_path):
    data = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.mp3'):
                # category = root.split('/')[-1]
                audio = file.split('.')[0]
                data[audio] = root + '/' + file
    return data

def pack_to_metadata(data, music_dir='Looperman_Loops/'):
    metadata = []
    real_audio_list = get_audio_list(music_dir)

    for category in data:
        for audio in tqdm(data[category]):
            category = category.replace(' ', '_')
            if audio in real_audio_list:
                real_path = real_audio_list[audio]
                if category != real_path.split('/')[-2]:
                    target_dir = real_path.replace(real_path.split('/')[-2], category)
                    os.rename(real_path, target_dir)
                    
                # file = f"Looperman_Loops/{category.replace(' ', '_')}/{audio}.mp3"
                info = data[category.replace("_", " ")][audio]
                temp = {
                    'audio': real_path,
                    'category': category,
                    'bpm': info['bpm'],
                    'genre': info['genre'],
                    'key': info['key'],
                    'description': info['description'],
                    'url': info['url']
                }
                metadata.append(temp)

    return metadata

def append_to_jsonl(data, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            pass
    if data:
        with open(file_path, 'a', encoding='utf-8') as file:
            json.dump(data, file)
            file.write('\n')

def load_model(file_path):
    model_path = file_path
    model, utils = torch.hub.load(repo_or_dir=model_path,
                                  model='silero_vad', source='local')
    device = torch.device('cuda')
    model.to(device)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    return model, device, utils

class MusicFilter:
    def __init__(self, filename, model, device, utils, snr_threshold=10):
        self.wav, self.sr = librosa.load(filename, sr=SAMPLING_RATE)
        self.thres = snr_threshold

        # VAD
        wav = (self.wav - np.min(self.wav)) / (np.max(self.wav) - np.min(self.wav))
        speech_timestamps = utils[0](torch.Tensor(wav).to(device), model, sampling_rate=SAMPLING_RATE)
        self.vad = len(speech_timestamps) == 0

        # SNR
        signal_power = np.square(self.wav).mean()
        noise = np.random.normal(0, 1, len(self.wav))
        noise_power = np.square(noise).mean()
        self.snr_value = -10 * np.log10(signal_power / noise_power)

    def get_results(self):
        return {
            "snr": self.snr_value, 
            "is_music": self.vad,
            "is_high_quality": self.snr_value >= self.thres,
            "valid": self.vad and self.snr_value >= self.thres
        }

    def __getattr__(self, item):
        if item == "snr":
            return self.snr_value
        elif item == "is_music":
            return self.vad
        elif item == "is_high_quality":
            return self.snr_value >= self.thres
        elif item == "valid":
            return self.vad and self.snr_value >= self.thres

def task1(): # pack the origin data to new metadata after comparing with the real audio list
    origin_file = "meta.json"
    output_file = "metadata.json"

    data = load_json(origin_file)
    metadata = pack_to_metadata(data)
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def task2(): # statistics about categories
    data = load_json("metadata.json")
    category_dict = {}
    for item in data:
        category = item['category']
        if category not in category_dict:
            category_dict[category] = 1
        else:
            category_dict[category] += 1

    sorted_dict = dict(sorted(category_dict.items(), key=lambda x: x[1], reverse=True))
    print(sorted_dict)


def task3(task_id=0, task_splits=10, cache_dir='cache'): # filter the music
    data = load_json("metadata.json")
    model_path = '/home/whatx/.cache/torch/hub/snakers4_silero-vad_master'
    model, device, utils = load_model(model_path)
    
    data_parts = [data[i::task_splits] for i in range(task_splits)]
    # print([len(item) for item in data_parts])
    task_list = data_parts[task_id]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for _, item in enumerate(tqdm(task_list)):
        
        mf = MusicFilter(item["audio"], model, device, utils, snr_threshold=10)
        cache = mf.get_results()
        temp = {
            "audio": item["audio"],
            "snr": cache["snr"],
            "is_music": 1 if cache["is_music"] else 0,
            "is_high_quality": 1 if cache["is_high_quality"] else 0,
            "valid": 1 if cache["valid"] else 0
        }

        append_to_jsonl(temp, f"{cache_dir}/task_{task_id}_of_{task_splits}.jsonl")

def task4(): # merge the cache files into one, update the info into metadata
    cache_dir = 'cache'
    output_file = 'sum.json'
    origin_file = 'metadata.json'

    cache_files = [f"{cache_dir}/{file}" for file in os.listdir(
        cache_dir) if file.endswith('.jsonl')]

    data = {}
    for file in cache_files:
        with open(file, 'r') as f:
            for line in f:
                temp = json.loads(line)
                idx = temp["audio"].split('/')[-1].split('.')[0]
                data[idx] = {
                    "snr": temp["snr"],
                    "is_music": temp["is_music"],
                    "is_high_quality": temp["is_high_quality"],
                    "valid": temp["valid"]
                }

    with open(os.path.join(cache_dir, output_file), 'w') as f:
        json.dump(data, f, indent=4)

    metadata = load_json(origin_file)
    for item in tqdm(metadata):
        audio = item["audio"]
        idx = audio.split('/')[-1].split('.')[0]
        if idx in data:
            item["snr"] = data[idx]["snr"]
            item["is_music"] = data[idx]["is_music"]
            item["is_high_quality"] = data[idx]["is_high_quality"]
            item["valid"] = data[idx]["valid"]
            item["duration"] = librosa.get_duration(filename=audio)
        else:
            print(f"Missing: {audio}")

    temp_file = "_" + origin_file
    with open(temp_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    os.rename(temp_file, origin_file)

def task5(): # check & stats
    snr_threshold = 10
    sep_char = "="*100
    data = load_json("metadata.json")
    missing = total_hour = aft_filter_hour = 0
    category_dict = {}
    aft_filter_dict = {}
    aft_duration_lst = []
    new_data = []
    for item in data:
        # check if have the "snr" key
        if "snr" not in item:
            missing += 1
        # stats
        category = item['category']
        total_hour += item["duration"] / 3600

        if category not in category_dict:
            category_dict[category] = 1
        else:
            category_dict[category] += 1

        if item["snr"] > snr_threshold:
            aft_filter_hour += item["duration"] / 3600
            if category not in aft_filter_dict:
                aft_filter_dict[category] = 1
            else:
                aft_filter_dict[category] += 1
            aft_duration_lst.append(item["duration"])
            new_data.append(item)

    # print(f"Missing: {missing}")
    category_dict_ = dict(sorted(category_dict.items(), key=lambda x: x[1], reverse=True))
    aft_filter_dict_ = dict(sorted(aft_filter_dict.items(), key=lambda x: x[1], reverse=True))
    print(f"Valid: {len(new_data)}/{len(data)} records")
    print(f"Valid Hour: {aft_filter_hour:.2f}/{total_hour:.2f} hours")
    print(sep_char, f"\nThe snr threshold is {snr_threshold}\n{sep_char}")
    print(f"Category:\n{category_dict_}", f"\n{sep_char}")
    print(f"Valid Category:\n{aft_filter_dict_}\n{sep_char}")
    print(f"Valid Duration: {np.mean(aft_duration_lst):.2f} +/- {np.std(aft_duration_lst):.2f} seconds")

    with open("metadata_snr10.json", 'w') as f:
        json.dump(new_data, f, indent=4)

def task6(): # sample metadata for three splits of 10 hours

    def sample(data, target_hour, n_split):
        splits = [[] for i in range(n_split)] # store the n_split data lists
        cate_stats = [{} for i in range(n_split)] # store the category stats dicts
        total_hour = [0] * n_split # store the total hour of each split
        total_cate = {}

        # sort data by category counts
        cate_count = Counter([item["category"] for item in data])
        cate_num = len(cate_count)
        sorted_data = sorted(
            data, key=lambda x: cate_count[x["category"]], reverse=False)
    
        for num, item in enumerate(tqdm(sorted_data)):
            category = item["category"]
            if category not in total_cate:
                total_cate[category] = 1
            else:
                total_cate[category] += 1

            for i in range(n_split):
                ave_num = int((
                    target_hour - total_hour[i]) * (3600/15) // (cate_num - len(cate_stats[i]))) if cate_num > len(cate_stats[i]) else math.inf
                
                if num % n_split == i:
                    if cate_stats[i].get(category, 0) > ave_num:
                        continue

                    if total_hour[i] <= target_hour:
                        if category not in cate_stats[i]:
                            cate_stats[i][category] = 1
                        else:
                            cate_stats[i][category] += 1

                        splits[i].append(item)
                        total_hour[i] += item["duration"] / 3600
                        data.remove(item)
            
        # print(total_cate)
        return data, splits, total_hour, cate_stats

    n_split = 3
    target_hour = 10
    data = load_json("metadata_snr10.json")
    data, splits, total_hour, cate_stats = sample(data, target_hour, n_split)

    stats = []
    for i in range(n_split):
        with open(f"metadata_snr10_{i}.json", 'w') as f:
            json.dump(splits[i], f, indent=4)
        stats.append({
            "split": i,
            "total_hour": total_hour[i],
            "category_stats": cate_stats[i]
        })

    with open(f"metadata_snr10_stats.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    with open(f"metadata_snr10_remain.json", 'w') as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task_id", type=int, default=0, required=True)
    parser.add_argument("--task_splits", type=int, default=10)
    args = parser.parse_args()

    # task1()
    # task2()
    # task3(args.task_id, args.task_splits)
    # task4()
    # task5()
    task6()

# if __name__ == "__main__": # multi-process
#     import multiprocessing
#     pool = multiprocessing.Pool(10)
#     for i in range(10):
#         pool.apply_async(task3, args=(i, 10))
#     pool.close()
#     pool.join()

if __name__ == "__main__": # single process
    # cd ~/AudioSet/MUEdit && CUDA_VISIBLE_DEVICES=0 python prepare.py --task_id 0 --task_splits 10
    # cd ~/AudioSet/MUEdit && CUDA_VISIBLE_DEVICES=1 python prepare.py --task_id 1 --task_splits 10
    main()
