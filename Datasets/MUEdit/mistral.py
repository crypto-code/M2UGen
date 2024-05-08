import torch, warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM, LlamaTokenizer
warnings.filterwarnings("ignore")

def load_model(model_path, device="cuda", model_type="Instruct"):
    if model_type == "instruct": # Load the mistral-instruct-v0.2 model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            pad_token_id=tokenizer.eos_token_id
        )
        print("Load the mistral-instruct-v0.2 model.")

    elif model_type == "base": # Load the mistral-base-v0.2 model
        model = MistralForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        print("Load the mistral-base-v0.2 model.")

    device = torch.device(device)
    model = model.to(device)

    return model, tokenizer

# Function to generate text
def generate_text(
        model, tokenizer, device, prompt, 
        max_length=1024, 
        temperature=1.9, # 0.5
        top_p=0.9, 
        top_k=50, # 40
        num_return_sequences=1, 
        max_new_tokens=128
    ):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            # max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            # attention_mask=input_ids.ne(tokenizer.pad_token_id).long()
        )
    
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    return generated_text

def prompt1(text):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You only output one sentence following the instructions:\n"
            "1.Remove all the Chinese characters of the text and bracket annotation.\n"
            "2.Rephrase the remain sentence.\nText: {text}")).format(text=text)
    )
    return final_prompt

def prompt2(source, target):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to tell a model how to implement the music editing operation and only output one sentence about how to format the instruction from the source music to target music:\n"
            "Source music description: {source}\n"
            "Target music description: {target}")).format(source=source, target=target))
    return final_prompt

def prompt3(source, target):
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
        )).format(source=source, target=target))
    return final_prompt


def prompt_instr(text):
    final_prompt = "[INST] {content} [/INST]".format(content = text)
    return final_prompt

def prompt_re(text):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You need to rephrase the following text without changing the original semantics:\n"
            "Text: {text}")).format(text=text))
    return final_prompt

def prompt_base(text):
    return text

def parse_result(generated_text, model_type, text):
    if model_type == "instruct":
        try:
            question = generated_text[0].split("[/INST]")[0].split("[INST] ")[1]
            answer = generated_text[0].split("[/INST] ")[1]
        except:
            return None, None
    elif model_type == "base":
        try :
            answer = generated_text[0].split(text + " ")[1]
            return None, answer
        except: # if the model does not generate any text, return None
            answer = generated_text
            return None, answer
            
    return question, answer

def process(model, tokenizer, device, text, prompt=prompt2, model_type="Instruct"):
    if prompt in [prompt2, prompt3]:
        final_prompt = prompt(text[0], text[1])
    else:
        final_prompt = prompt(text)

    generated_text = generate_text(model, tokenizer, device, final_prompt)
    if model_type == "instruct":
        question, answer = parse_result(generated_text, model_type, text)
    elif model_type == "base":
        question = text
        _, answer = parse_result(generated_text, model_type, text)
    return question, answer

def main():
    # 1.Load the model and tokenizer
    model_type = "instruct"
    path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf" # instruct model
    device = "cuda"
    sep = 90
    model, tokenizer = load_model(model_path=path, device=device, model_type=model_type)
    # 2.Set the model to evaluation mode
    model.eval()
    # 3.Define the prompt & generate text
    # text = "Slow and romantic music, such as classical music or jazz, is suitable for a romantic dinner. [slow]\n（中文翻译：Q4. 什么样的音乐适合浪漫的晚餐？\nA4. 缓慢而浪漫的音乐，如古典音乐或爵士乐，适合浪漫的晚餐。【慢】）"

    # 1) Speed part
    text = ['This music covers 0.7x time in speed.', 'This music covers 1.5x time in speed with the same pitch.']
    # text = 'To achieve a 1.5x faster tempo while maintaining the original pitch in the music editing process, adjust the tempo setting accordingly.'
    # rephrase --> 'To increase the tempo by 50% while keeping the pitch unchanged during music editing, make the necessary adjustments to the tempo setting. '

    # 2) Pitch part
    # text = ['This music is in original pitch.', 'This music is 1 semitones higher in pitch.']
    # text = ['This music is in original pitch.', 'This music is 500 cents higher in pitch.']
    # text = 'To elevate the original pitch of the source music by one semitone for the target music, apply a pitch shift increase of 1 semitone.'
    # rephrase --> 'To increase the musical note's pitch by one semitone for the altered version, apply a pitch shift filter with a value of +1 semitone.'
    # text = 'To produce the desired music, increase the pitch of the original music by 500 cents.'
    # rephrase --> 'To achieve the desired musical tone, raise the pitch of the original melody by approximately one and a half tones (500 cents).'
    
    # text = "What's wrong with this sentence? Sentence: 'Raise the pitch of the input music by a semi-tones equivalent to 500 cents.'"
    question, answer = process(model, tokenizer, device, text, prompt=prompt2, model_type=model_type)
    print(f"{'='*sep}\n{question}\n{'='*sep}")
    print(answer, f"\n{'='*sep}")

if __name__ == "__main__":
    main()