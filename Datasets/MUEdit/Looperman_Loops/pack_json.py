#############################################################################
import json
from tqdm import tqdm
from config import *
import os

def pack_json():
    meta = {}
    for category in sorted_categories:
        meta[category['name']] = {}
        for page in tqdm(range(1, category['max_page'] + 1)):
            file_name = "{0}_page_{1}.json".format(category['name'], page)
            file_path = raw_path + file_name

            with open(file_path, 'r') as file:
                data = json.load(file)
                for key, value in data[category['name']].items():
                    meta[category['name']][key] = value
                print("Packing {0} is done!".format(file_name))
    with open(output_path + 'meta.json', 'w') as f:
        f.write(json.dumps(meta, indent=4))
    return meta

def read_json(file):
    with open(file, 'r') as f:
        meta = json.load(f)
    return meta

def get_download_target(meta_file):
    meta = read_json(meta_file)
    download_dic = {}
    for category in sorted_categories:
        download_dic[category['name']] = []
        for key, _ in meta[category['name']].items():
            url = "https://www.looperman.com/media/loops/{1}/{0}.mp3".format(key, key.split("-")[2])
            download_dic[category['name']].append(url)

    # write the download dictionary into a file
    with open(os.path.join(output_path, 'download_dic.json'), 'w') as f:
        f.write(json.dumps(download_dic, indent=4))
    
    # write the summary of the download dictionary into a file
    with open(os.path.join(output_path, 'download_summary.txt'), 'w') as f:
        print(["{0}:{1}".format(i,len(j)) for i,j in download_dic.items()])
        print("There are {} music files to be downloaded.".format(sum([len(j) for _,j in download_dic.items()])))
        f.write("There are {} music files to be downloaded.\n\n".format(sum([len(j) for _,j in download_dic.items()])))
        for i,j in download_dic.items():
            f.write("{0}:{1}\n".format(i,len(j)))

    return download_dic

def download_music(checkpoint_file, download_dic):
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        start_category_index = checkpoint['category_index']
        start_url_index = checkpoint['url_index']
    except (FileNotFoundError, json.JSONDecodeError):
        # if the checkpoint file does not exist, start from the beginning
        start_category_index = 0
        start_url_index = 0

    for category_idx, category in enumerate(sorted_categories[start_category_index:]):
        category_idx += start_category_index
        if not os.path.exists(output_path + category['name']):
            os.makedirs(output_path + category['name'])
        path = output_path + category['name']
        for idx, url in enumerate(download_dic[category['name']][start_url_index:]):
            idx += start_url_index
            os.system("wget -P {0} {1}".format(path, url))
            # save checkpoint for the music download
            checkpoint = {
                'category_index': category_idx,
                'url_index': idx
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

# return the missing files dictionary like download_dic
def get_missing_dict():
    # find the current music files
    current_dict = {}
    for file in os.listdir(output_path):
        sub_path = os.path.join(output_path, file)
        if os.path.isfile(sub_path):
            pass
        else:
            current_dict[file] = []
            for music in os.listdir(sub_path):
                print(music)
                if music.endswith('.mp3'):
                    key = music.split(".")[0]
                    url = "https://www.looperman.com/media/loops/{1}/{0}.mp3".format(key, key.split("-")[2])
                    current_dict[file].append(url)
    # compare the current music files with the download_dic and find out the missing files
    missing_dic = {}
    for key, value in download_dic.items():
        missing_dic[key] = []
        if key not in current_dict:
            continue
        for url in value:
            if url not in current_dict[key]:
                missing_dic[key].append(url)
    
    # write the missing dictionary into a file
    with open(os.path.join(output_path, 'missing_dic.json'), 'w') as f:
        f.write(json.dumps(missing_dic, indent=4))

    return missing_dic


if __name__ == '__main__':
    # 1.pack the json file into meta.json
    """
    meta = pack_json()
    print("Packing json file is done!")
    """

    # 2.set the download dictionary
    meta_file = output_path + 'meta.json'
    download_dic = get_download_target(meta_file=meta_file)

    # 3.load the checkpoint and download the mp3 file
    # checkpoint_file = checkpoint_path + 'music_ckpt.json'
    # download_music(checkpoint_file=checkpoint_file, download_dic=download_dic)

    # 4.check the music files and find out the missing files
    missing_dic = get_missing_dict()

    # 5.download the missing files
    checkpoint_file = checkpoint_path + 'missing_ckpt.json'
    download_music(checkpoint_file=checkpoint_file, download_dic=missing_dic)
