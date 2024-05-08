import os, sys, json, random, argparse, time
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MUEdit"))
from mistral import *
from tqdm import tqdm
import numpy as np

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def append_to_jsonl(data, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            pass
    if data:
        with open(file_path, 'a', encoding='utf-8') as file:
            json.dump(data, file)
            file.write('\n')

def prompt_response(music, vision, tags, modality="image"):
    final_prompt = "[INST] {content} [/INST]".format(
        content = (
            f"Follow these steps and generate a response according to the descriptions of the Music, its {modality.capitalize()} and Tags:\n"
            f"1.Your response should start with: 'Here is a music for the {modality} that is ...'\n"
            "2.You should describe main instruments in the music in terms of the instruments in Tags.\n"
            # "3.Additional information could refer to the description of Music and Image.\n"
            f"3.Refer to the {modality} when describing the music.\n"
            f"4.You need to complete the response given the Music, {modality.capitalize()} and Tags.\n"
            f"5.Output the response incorporating the {modality.capitalize()} into the Music.\n\n"
            f"Music: {music}\n\n{modality.capitalize()}: {vision}\n\nTags: {tags}"
    ))
    return final_prompt

def prompt_instruction(num_words=20, modality="image"):
    return "[INST] {content} [/INST]".format(
        content = (
            f"Rephrase the instruction within {num_words} words, output should still start with 'Generate'\n"
            f"Instruction: Generate a music to match the {modality}."
    ))

def denoise_tags_list(tags):
    if 'music' in tags:
        tags.remove('music')
    if 'musical instrument' in tags:
        tags.remove('musical instrument')
    tags = ", ".join(tags)
    return tags

def calculate_time(sep=90):
    def wrapper(func):
        def inner(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Execution time: {end-start:.2f} seconds")
            print(f"{'='*sep}\n")
            return result
        return inner
    return wrapper

@calculate_time(sep=90)
def qa(text, model_dict, sep):
    question, answer = process(
        model=model_dict["model"], 
        tokenizer=model_dict["tokenizer"], 
        device=model_dict["device"],
        model_type=model_dict["model_type"],
        prompt=prompt_base,
        text=text
    )
    print(f"{'='*sep}\n{question}\n{'='*sep}")
    print(answer, f"\n{'='*sep}\n")
    return question, answer

def construct_response(data, cache, sep, sub_id, split_num, model_dict, modality="image"):
    print(f"The total number of this {modality} subset split data is {len(data)}")

    for idx, item in enumerate(tqdm(data)):
        music = item["audio_caption"]
        vision = item.get(f"{modality}_caption")
        tags = denoise_tags_list(item["tags"])
        text = prompt_response(music, vision, tags, modality=modality)
        if os.path.exists(cache):
            cache_data = load_jsonl(cache)
            cache_list = [d["ytid"] for d in cache_data]
            if item["ytid"] in cache_list:
                continue
        _, a = qa(text, model_dict, sep)
        temp = {
            "ytid": item["ytid"],
            "response": a
        }
        print(f"Processing {idx+1}/{len(data)} of the {modality}-{sub_id} split out of {split_num}...")
        append_to_jsonl(temp, cache)

    # print(len(set(cache_list)))
    # data_lst = [sub["ytid"] for sub in data]
    # duplicates = [item for item in set([i for i in data_lst if data_lst.count(i) > 1])]
    # print(duplicates)
    # PS: These duplicates result from one record having overlapping tags with instruments types.

def construct_instruction(pool_path, sep, model_dict, iter_num=500):
    for modal in ["image", "video"]:
        for i in tqdm(range(iter_num)):
            _, a = qa(prompt_instruction(num_words=20, modality=modal), model_dict, sep)
            if a is None:
                continue
            temp = {
                "modality": modal,
                "instruction": a
            }
            append_to_jsonl(temp, pool_path)
            print(f"Processing {i+1}/{iter_num} of the {modal} instruction...")

def filter_instruction(pool_path):
    data = load_jsonl(pool_path)
    new_data = {"video": [], "image": []}
    out = 0
    for item in tqdm(data):
        place = item["modality"]
        instr = item["instruction"]
        if instr.startswith("Generate") and (not '\n' in instr):
            if not any(char.isdigit() for char in instr) and not any(char in instr for char in "()"):
                new_data[place].append(instr)
        else:
            out += 1
            print(f"Filtered instruction: {instr}")
    save_json(new_data, pool_path.replace(".jsonl", ".json"))
    print(f"Filter out {out} instructions.")
    return new_data

def get_merge_response(cache_dir):
    targets = ["image", "video"]
    
    merge_data = {"image": {}, "video": {}}
    for target in tqdm(targets):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith(f"cache_{target}_")]
        for file in cache_files:
            data = load_jsonl(os.path.join(cache_dir, file))
            for item in data:
                ytid = item["ytid"]
                response = item["response"]
                if ytid not in merge_data[target]:
                    merge_data[target][ytid] = response
    return merge_data

def merge_sft_data(merge_response, instr_data):
    final_data = {"image": [], "video": []}
    for target in final_data.keys():
        metadata = load_json(f"metadata_{target}.json")
        for item in metadata:
            ytid = item["ytid"]
            input_file = item[f"{target}"]
            input_modality = f"{target}"
            input_caption = item[f"{target}_caption"]
            output_file = item["audio"]
            output_modality = "audio"
            output_caption = item["audio_caption"]

            instr = random.choice(instr_data[target])
            response = merge_response[target][ytid]

            temp = {
                "input_file": input_file,
                "output_file": output_file,
                "conversations": [
                    {
                        "from": "human",
                        "value": instr,
                        "input_modality": input_modality,
                        "caption": input_caption
                    },
                    {
                        "from": "gpt",
                        "value": response,
                        "output_modality": output_modality,
                        "caption": output_caption
                    }
                ]
            }
            final_data[target].append(temp)

        save_json(final_data[target], f"MU{target.capitalize()}_Instructions.json")
        print(f"Save {target} data into MU{target.capitalize()}_Instructions.json")
    return final_data

def main():
    # cd ~/AudioSet/MUVideo && CUDA_VISIBLE_DEVICES=0 python caption.py --split image --split_num 5 --sub_id 0
    # cd ~/AudioSet/MUVideo && CUDA_VISIBLE_DEVICES=1 python caption.py --split video --split_num 5 --sub_id 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="which split would be used, video or image")
    parser.add_argument("--split_num", type=int, default=10, help="total number of each split")
    parser.add_argument("--sub_id", type=int, default=0, help="sub id of the split")
    args = parser.parse_args()
    split = args.split
    split_num = args.split_num
    sub_id = args.sub_id

    # load model
    sep = 90
    device = "cuda"
    model_type = "instruct"
    model_path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    pool_path = "vision_pool.jsonl"
    cache_image, cache_video = f"caption_cache/cache_image_{sub_id}.jsonl", f"caption_cache/cache_video_{sub_id}.jsonl"

    model, tokenizer = load_model(model_path, device=device, model_type=model_type)
    model.eval()

    model_dict = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "model_type": model_type,
        "sub_id": sub_id,
        "split_num": split_num
    }

    # task 1: construct response for image or video, put into cache_image or cache_video
    if args.split == "image":
        cache = cache_image
        image_data = load_json(f"metadata_{args.split}.json")
        subset_data = np.array_split(image_data, split_num)[sub_id]
        construct_response(subset_data, cache, sep, sub_id, split_num, model_dict, modality=split)
    elif args.split == "video":
        cache = cache_video
        video_data = load_json(f"metadata_{args.split}.json")
        subset_data = np.array_split(video_data, split_num)[sub_id]
        construct_response(subset_data, cache, sep, sub_id, split_num, model_dict, modality="video")
    else:
    # task 2: construct instructions for them, put into pool_path
        construct_instruction(pool_path, sep, model_dict)
        filter_instruction(pool_path)
    # task 3: merge information and get final SFT data for two splits (image and video)
        merge_response = get_merge_response("caption_cache")            # --> dict then dict
        instr_data = load_json(pool_path.replace(".jsonl", ".json"))    # --> dict then list
        merge_sft_data(merge_response, instr_data)


if __name__ == "__main__":
    main()