# CUDA_VISIBLE_DEVICES=0 python llava_caption.py --task_id 0
import json
import os
from tqdm import tqdm
import numpy as np
from llava.eval.run_llava import *
from argparse import ArgumentParser

prompt = (""
    "Describe the image within one sentence following these instructions:\n"
    "1.Recognize all the instruments that appear in this image.\n"
    "2.Detect all of other elements which may contribute to the music effect.\n"
    "3.Other elements may include Facial Expression and Body Language, Mood, Atmosphere, "
    "and Lighting, Environment, Clothing, Color Scheme and Composition, Audience Interaction, Symbolism.\n"
    "4.Combine all the elements above and highlight the instruments into the sentence.\n"
    "5.Final output should be limited to 150 words, formatted as a single paragraph, expressed in one to three sentences.")

default_load_args = {
    "model_path": "../llava-v1.6-34b",
    "model_base": None,
    "load_8bit": False,
    "load_4bit": True,
    "use_flash_attn": False,
}

default_infer_args = {
    "image_files": [""],
    "query": "",
    "temperature": 1,
    "top_p": None,
    "num_beams": 1,
    "device": "cuda",
    "max_new_tokens": 256,
}

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

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

def load_model(load_args):
    disable_torch_init()

    model_name = get_model_name_from_path(load_args["model_path"])
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=load_args["model_path"], 
        model_base=load_args["model_base"],
        model_name=model_name, 
        load_8bit=load_args["load_8bit"], 
        load_4bit=load_args["load_4bit"],
        use_flash_attn=load_args["use_flash_attn"],
    )
    return tokenizer, model, image_processor

def inference(infer_args, tokenizer, model, image_processor):
    
    start_time = time.time()
    # process query
    qs = infer_args["query"]
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # if "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # load and process images
    images = load_images(infer_args["image_files"])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    # generate input_ids
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(infer_args["device"])
    )

    # inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if infer_args["temperature"] > 0 else False,
            temperature=infer_args["temperature"],
            top_p=infer_args["top_p"],
            num_beams=infer_args["num_beams"],
            max_new_tokens=infer_args["max_new_tokens"],
            use_cache=True,
        )

    # decode output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    print("*****")
    print(f"Time: {time.time() - start_time}")
    return outputs

def split_task(data, num_splits=7):
    num_list = []
    image_list = []
    n = 0
    for num, v in enumerate(data):
        image = v['image']
        num_list.append(num)
        image_list.append(image)
        n += 1
    splits = np.array_split(np.array(num_list), num_splits)
    print(n)
    return splits

# Set the ckpt for the process_task by load the length of the jsonl
def load_ckpt(file_path):
    if not os.path.exists(file_path):
        return 0
    else:
        with open(file_path, "r") as f:
            lines = f.readlines()
            return len(lines)

def process_task(data, splits, task_id, load_args, infer_args, save_path=None):
    # Load the model
    tokenizer, model, image_processor = load_model(load_args=load_args)
    print("--------------------------------------------------")
    print(prompt)
    print("--------------------------------------------------")
    # Load the ckpt
    start_loc = load_ckpt(file_path=save_path)
    for num, v in tqdm(enumerate(data)):
        if num in splits[task_id] and num >= start_loc:
            image = v['image']
            infer_args["image_files"] = [image]
            infer_args["query"] = prompt
            image_caption = inference(
                infer_args=infer_args, 
                tokenizer=tokenizer, 
                model=model, 
                image_processor=image_processor
            )
            temp = {
                "image": image,
                "caption": image_caption
            }
            append_to_jsonl(data=temp, file_path=save_path)
            print(f"Task {task_id} - Image {num} - Done")
            print("--------------------------------------------------")
        elif num in splits[task_id] and num < start_loc:
            print(f"Task {task_id} - Image {num} - Skipped")
            print("--------------------------------------------------")

    return data

def main():
    parser = ArgumentParser()
    # Define the path
    input_path = "metadata.json"
    # output_path = "metadata_image_caption.json"
    # Define the args for the model loading
    load_args = {
        "model_path": "~/llava-v1.6-34b",
        "model_base": None,
        "load_8bit": False,
        "load_4bit": True,
        "use_flash_attn": False,
    }
    # Define the args for the model inference
    infer_args = {
        "image_files": [""],
        "query": "",
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        "device": "cuda",
        "max_new_tokens": 256,
    }
    # Define splits and task id, and output path
    num_splits = 7
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()
    output_path = f"caption_cache/image_caption_{args.task_id}.jsonl"

    # Start the process
    data = load_json(file_path=input_path)
    splits = split_task(data=data, num_splits=num_splits)
    process_task(
        data=data, 
        splits=splits, 
        task_id=args.task_id, 
        load_args=load_args, 
        infer_args=infer_args, 
        save_path=output_path
    )

    # save_json(data, file_path=output_path)

if __name__ == "__main__":
    main()