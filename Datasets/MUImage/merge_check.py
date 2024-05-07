import json
import os
from AudioSet.MUImage.llava_caption import *

def merge(input_path, output_path, subfile_dir):
    # Extract the filename end with .jsonl from the subfile_dir
    subfile_list = sorted(
        [os.path.join(subfile_dir, file) for file in os.listdir(subfile_dir) 
            if file.endswith(".jsonl")])
    
    # Load the jsonl file and check if it exists in the metadata.json
    data = load_json(input_path)
    for _, subfile in enumerate(subfile_list):
        sub_data = load_jsonl(subfile)
        for record in tqdm(sub_data):
            image = record['image']
            caption = record['caption']
            for num, v in enumerate(data):
                if image == v['image']:
                    data[num]['image_caption'] = caption
        print(f"File {subfile} - Done")
    
    # Save the metadata_merge.json
    save_json(data, output_path)

def check(check_path, new_path):
    data = load_json(check_path)
    total = len(data)
    no_caption = 0
    miss_caption = 0
    tokenizer, model, image_processor = load_model(load_args=default_load_args)
    print("--------------------------------------------------")
    default_infer_args["query"] = prompt
    print(prompt)
    print("--------------------------------------------------")
    
    for num, v in tqdm(enumerate(data)):
        image = v['image']
        
        if "image_caption" not in v:
            no_caption += 1

            default_infer_args["image_files"] = [image]
            image_caption = inference(
                infer_args=default_infer_args, 
                tokenizer=tokenizer, 
                model=model, 
                image_processor=image_processor
            )
            data[num]['image_caption'] = image_caption
            print("--------------------------------------------------")

        elif ("\n" in v['image_caption']) or (
            not v['image_caption'].endswith(".")):
            miss_caption += 1

            default_infer_args["image_files"] = [image]
            image_caption = "\n"
            while ("\n" in image_caption) or (not image_caption.endswith(".")):
                image_caption = inference(
                    infer_args=default_infer_args, 
                    tokenizer=tokenizer, 
                    model=model, 
                    image_processor=image_processor
                )
            data[num]['image_caption'] = image_caption
            print("Finish the {} missing caption.".format(miss_caption))
            print("--------------------------------------------------")

    print(f"Total: {total}")
    print(f"No caption: {no_caption}")
    print(f"Miss caption: {miss_caption}")
    save_json(data, new_path)


def main():
    # Define the path
    input_path = "metadata.json"
    output_path = "metadata_v2.json" # metadata_merge.json, metadata_v1.json 
    subfile_dir = "caption_cache"
    new_path = "metadata_v3.json" # metadata_v1.json, metadata_v2.json

    # 1.Merge the caption to the metadata.json
    # merge(input_path=input_path, output_path=output_path, subfile_dir=subfile_dir)

    # 2.Check the caption
    check(check_path=output_path, new_path=new_path)

if __name__ == "__main__":
    main()