import os, sys, json, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def prompt_speed1_text(src_time, tgt_time):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to follow these steps given description of input and target music:\n"
            "1.Imagine you are a human who want to modify the music using the model.\n"
            "2.Provide an instruction on converting input music to target music.\n"
            "3.You'd better not to mention input and target music in the instruction.\n"
            "4.You could describe the music using numbers or coarse degrees.\n"
            "5.Output one sentence within 20 words.\n"
            "Input music description: {source}\n"
            "Target music description: {target}"
        )).format(
            source="The music is in original speed." if src_time == 1.0 else random.choice([
                    f"The music covers {src_time}x time in speed.", 
                    "The music is in {:.1f}x speed.".format(1 / src_time)
            ]),
            target="The music is in original 1x speed." if tgt_time == 1.0 else random.choice([
                    f"The music covers {tgt_time}x time in speed with the same pitch.",
                    "The music is in {:.1f}x speed.".format(1 / tgt_time)
            ])
    ))
    return final_prompt

def prompt_speed2_text(src_time, tgt_time):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to rephrase the following text without changing the original semantics:\n"
            "Text: {text}")).format(
                text="{adjust} the input music to {operation}.".format(
                    adjust="Speed up" if src_time > tgt_time else "Slow down",
                    operation="{:.1f}x speed".format(src_time / tgt_time))
    ))
    return final_prompt

def prompt_speed3_text(src_time, tgt_time):
    degree = {
        0.5: "faster significantly",
        0.7: "faster slightly",
        1.3: "slower slightly",
        1.5: "slower significantly"
    }
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to rephrase the following text without changing the original semantics:\n"
            "Text: {text}")).format(
                text="{adjust} the {speed} of input music {operation}.".format(
                    adjust=random.choice(["Adjust", "Change", "Modify"]),
                    speed=random.choice(["speed", "tempo", "rhythm"]),
                    operation=degree[tgt_time])
    ))
    return final_prompt


def generate_human_side(sep, iter_num, model, tokenizer, device, model_type, temp_path):
    speed_ref = {"0_5": 0.5, "0_7": 0.7, "1_0": 1.0, "1_3": 1.3, "1_5": 1.5}
    caption_pool = []
    # for source, src_time in {"1_0": 1.0}.items():
    for prompt in [prompt_speed1_text, prompt_speed2_text, prompt_speed3_text]:
        source, src_time = "1_0", 1.0

        targets = {key: value for key, value in speed_ref.items() if key != source}
        for target, tgt_time in targets.items():

            for _ in tqdm(range(iter_num)):
                operation = "{:.1f}x speed".format(src_time / tgt_time)
                text = prompt(src_time, tgt_time)
                question, answer = process(
                    model, tokenizer, device, text, prompt=prompt_base, model_type=model_type)
                
                print(f"{'='*sep}\n{question}\n{'='*sep}")
                print(answer, f"\n{'='*sep}\n")
                temp = {
                    "source": src_time, 
                    "target": tgt_time,
                    "operation": operation,
                    "instruction": answer}
               
                caption_pool.append(temp)
                append_to_jsonl(temp, temp_path)
    print("Finished generating human side captions.")

def prompt_gpt_text(instruction):
    final_prompt = "[INST] {content} [/INST]".format(
        content = (
            "You will follow these steps and give an answer in terms of the text:\n"
            "1.You answer should start with: 'Here is a music that is ...'\n"
            "2.You need to respond to the instruction given in the text.\n"
            f"3.Output the answer in a single sentence.\nText: {instruction}"
    ))
    return final_prompt

def generate_gpt_side(sep, model, tokenizer, device, model_type, temp_path, pool_path):
    data = load_jsonl(temp_path)
    ref = {0.5: "0_5", 0.7: "0_7", 1.0: "1_0", 1.3: "1_3", 1.5: "1_5"}
    final_pool = {}
    for item in tqdm(data):
        target = ref[item["target"]]
        operation = item["operation"]
        instruction = item["instruction"]

        if target not in final_pool:
            final_pool[target] = {}
            final_pool[target]["operation"] = operation
            final_pool[target]["pairs"] = []

        text = prompt_gpt_text(instruction)
        question, answer = process(
            model, tokenizer, device, text, prompt=prompt_base, model_type=model_type)
        item["gpt"] = answer
        print(f"{'='*sep}\n{question}\n{'='*sep}")
        print(answer, f"\n{'='*sep}\n")
        
        temp = {"human": instruction, "gpt": answer}
        final_pool[target]["pairs"].append(temp)
    print("Finished generating GPT side captions.")
    os.remove(temp_path)
    save_json(final_pool, pool_path)

def construct_MUEdit_speed_instructions(pool_path, metadata_path, final_path):
    pool = load_json(pool_path)
    metadata = load_json(metadata_path)

    final_data = []
    file_list = [item["audio"].split("/")[-1] for item in metadata]
    group_len = len(pool.keys())
    splits = np.array_split(file_list, group_len)

    for num, (dir_name, value) in enumerate(tqdm(pool.items())):
        subpool = value["pairs"]
        for file_name in splits[num]:
            sample = random.choice(subpool)
            output_file = os.path.join(dir_name, file_name)
            temp = {
                "input_file": os.path.join("1_0", file_name),
                "output_file": output_file,
                "conversation": [
                    {
                        "from": "human",
                        "value": sample["human"],
                        "input_modality": "audio"
                    },
                    {
                        "from": "gpt",
                        "value": sample["gpt"],
                        "output_modality": "audio"
                    }
                ]
            }
            final_data.append(temp)
    
    save_json(final_data, final_path)
    print("Finished constructing MUEdit Speed Instructions.")
    print(f"Total number of instructions: {len(final_data)}")
    print(f"Here are number for each splits: {[len(value) for value in splits]}")

def main():
    sep = 90
    iter_num = 25

    device = "cuda"
    model_type = "instruct"
    model_path = "/home/whatx/SusGen/ckpts/Mistral-7B-Instruct-v0.2-hf"
    temp_path = "speed_temp.jsonl"
    pool_path = "speed_pool.json"
    metadata_path = "../metadata_snr10_0.json"
    final_path = "MUEdit_Speed_Instructions.json"

    # load model for task1 and task2
    # model, tokenizer = load_model(model_path, device=device, model_type=model_type)
    # model.eval()

    # task1: generate human side captions
    # generate_human_side(sep, iter_num, model, tokenizer, device, model_type, temp_path)

    # task2: generate gpt side captions
    # generate_gpt_side(sep, model, tokenizer, device, model_type, temp_path, pool_path)
    
    # task3: construct MUEdit Speed Instructions
    construct_MUEdit_speed_instructions(pool_path, metadata_path, final_path)

if __name__ == "__main__":
    main()