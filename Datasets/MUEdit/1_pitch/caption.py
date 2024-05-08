import os, sys, json, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mistral import *
from tqdm import tqdm
import numpy as np

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl(file_path):
    try:
        return load_json(file_path)
    except:
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

def prompt_pitch1_text(change):
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
            source="The music is in original pitch.",
            target="The music is {0} {1} in pitch.".format(
                f"{abs(change)} cents",
                "higher" if change > 0 else "lower"
            )
    ))
    return final_prompt

def prompt_pitch2_text(change):
    if abs(change) == 100:
        operation = random.choice(["semi-tone", "semitone", "100 cents"])
    elif abs(change) == 200:
        operation = random.choice(["tone", "200 cents"])

    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to rephrase the following text without changing the original semantics:\n"
            "Text: {text}")).format(
                text="{adjust} the tone of input music by {operation}.".format(
                    adjust="Raise" if change > 0 else "Lower", operation=operation)
    ))
    return final_prompt

def prompt_pitch3_text(change):
    degree = {
        200: "higher significantly",
        100: "higher slightly",
        -100: "lower slightly",
        -200: "lower significantly"
    }
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to rephrase the following text without changing the original semantics:\n"
            "Text: {text}")).format(
                text="{adjust} the {pitch} of input music {operation}.".format(
                    adjust=random.choice(["Adjust", "Change", "Modify"]),
                    pitch=random.choice(["pitch", "tone"]),
                    operation=degree[change])
    ))
    return final_prompt


def generate_human_side(sep, iter_num, model, tokenizer, device, model_type, temp_path):
    pitch_ref = {"down_100": -100, "down_200": -200, "original": 0, "up_100": 100, "up_200": 200}
    caption_pool = []

    for prompt in [prompt_pitch1_text, prompt_pitch2_text, prompt_pitch3_text]:
        source, src_tone = "original", 0

        targets = {key: value for key, value in pitch_ref.items() if key != source}
        for target, tgt_tone in targets.items():

            for _ in tqdm(range(iter_num)):
                change = tgt_tone - src_tone
                operation = f"{change} cents"
                text = prompt(change)
                question, answer = process(
                    model, tokenizer, device, text, prompt=prompt_base, model_type=model_type)
                
                print(f"{'='*sep}\n{question}\n{'='*sep}")
                print(answer, f"\n{'='*sep}\n")
                temp = {
                    "source": src_tone, 
                    "target": tgt_tone,
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
    ref = {0: "original", -100: "down_100", -200: "down_200", 100: "up_100", 200: "up_200"}
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

def denoise_temp(temp_path):
    data = load_jsonl(temp_path)
    # if record contains '\n', remove the record
    for item in data:
        if "\n" in item["instruction"]:
            data.remove(item)
    with open(temp_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def construct_MUEdit_pitch_instructions(pool_path, metadata_path, final_path):
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
    model_path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    temp_path = "pitch_temp.jsonl"
    pool_path = "pitch_pool.json"
    metadata_path = "../metadata_snr10_1.json"
    final_path = "MUEdit_Pitch_Instructions.json"

    # load model for task1 and task2
    # model, tokenizer = load_model(model_path, device=device, model_type=model_type)
    # model.eval()

    # task1: generate human side captions
    # generate_human_side(sep, iter_num, model, tokenizer, device, model_type, temp_path)
    # denoise_temp(temp_path)

    # task2: generate gpt side captions
    # generate_gpt_side(sep, model, tokenizer, device, model_type, temp_path, pool_path)

    # task3: construct MUEdit pitch instructions
    construct_MUEdit_pitch_instructions(pool_path, metadata_path, final_path)

if __name__ == "__main__":
    main()