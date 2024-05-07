import torch, json, warnings, os, random
from tqdm import tqdm
warnings.filterwarnings("ignore")
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

inp = ("[INST] "
    "Describe the video within one sentence following these instructions:\n"
    "1.Recognize all the dynamic changes influences music tempo and intensity.\n"
    "2.Analyze how storyline progression and scene transitions dictate music mood and atmosphere.\n"
    "3.Examine how visual cues like beat changes and scene transitions guide music rhythm and arrangement.\n"
    "4.Combine all above, and finally output should be limited to 150 words, formatted as a single paragraph, expressed in one to three sentences."
    " [/INST]")

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

def single_caption(prompt, video, model, processor, tokenizer):
    inp = prompt
    video_processor = processor['video']

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    video_tensor = video_processor(video, video_decode_backend='pytorchvideo', return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    # print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    result = outputs.split("</s>")[0]

    return result

def main():
    disable_torch_init()
    global inp
    # video = 'audioset_videos/accordion_video/--WKuD_46Fk.mp4'
    video = "audioset_videos/accordion_video/-1qyr-IzqKY.mp4"
    video1 = '/home/whatx/AudioSet/audioset_videos/accordion_video/_5n9bckN6zI.mp4'
    model_path = '/home/whatx/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, 
                                load_8bit, load_4bit, device=device, cache_dir=cache_dir)

    print("=" * 50)
    result = single_caption(inp, video, model, processor, tokenizer)
    print(result)
    print("=" * 50)
    result1 = single_caption(inp, video1, model, processor, tokenizer)
    print(result1)
    print("=" * 50)


def test(lower_limit=0, upper_limit=5000):
    # 1.intialize the model
    disable_torch_init()
    global inp

    model_path = '~/Video-LLaVA-7B'
    cache_dir = '~/cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, 
            load_8bit, load_4bit, device=device, cache_dir=cache_dir)

    # 2. load the metadata
    data = load_json('metadata_v3.json')
    data = data[lower_limit:upper_limit]

    for num, item in enumerate(data):

        ytid = item['ytid']
        video = os.path.join("..", item['video'])

        # 3. process the video
        print(f"Processing {num + lower_limit} - {video}")
        result = single_caption(inp, video, model, processor, tokenizer)
        print(result, "\n", "=" * 50)

        temp = {
            'ytid': ytid,
            'video': video,
            'video_caption': result,
        }
        append_to_jsonl(temp, f'caption_cache/video_caption_{upper_limit//7500}.jsonl')

def merge_to_meta(input_file, caption_cache_dir, output_file):
    data = load_json(input_file)
    caption_cache = os.listdir(caption_cache_dir)
    for file in caption_cache:
        if file.endswith('.jsonl'):
            sub_data = load_jsonl(os.path.join(caption_cache_dir, file))
            for _, record in enumerate(tqdm(sub_data)):
                ytid = record['ytid']

                for num, item in enumerate(data):
                    if ytid == item['ytid']:
                        data[num]['video_caption'] = record['video_caption']
        print(f"File {file} - Done", '\n', "=" * 50)

    save_json(data, output_file)
    print(f"Saving the metadata {input_file} to {output_file}")

def sample_and_split(input_file, output_image, output_video, stats_file):
    data = load_json(input_file)
    audio_dict, put_state, image_dict, video_dict = {}, {}, {}, {}
    image_data, video_data = [], []
    for _, item in enumerate(tqdm(data)):
        audio = item['audio']
        audio_type = audio.split('/')[1].split('/')[0]
        if audio_type not in audio_dict:
            audio_dict[audio_type] = 1
            image_dict[audio_type] = 0
            video_dict[audio_type] = 0
            put_state[audio_type] = 0
        else:
            audio_dict[audio_type] += 1

        if put_state[audio_type] == 0:
            image_data.append(item)
            image_dict[audio_type] += 1
            put_state[audio_type] = 1
        else:
            video_data.append(item)
            video_dict[audio_type] += 1
            put_state[audio_type] = 0
        
    stats = []
    name = ['Balanced_AudioSet', 'MUImage', 'MUVideo']
    prev_list = [audio_dict, image_dict, video_dict]
    for num, prev in enumerate(prev_list):
        total = sum(prev.values())
        print(prev, '\nTotal: ', total, '\n', '=' * 50)
        stats.append({
            'name': name[num],
            'stats': prev,
            'total': total,
            'hours': "{:.2f} hours".format(10 * total / 3600)
        })

    save_json(image_data, output_image)
    save_json(video_data, output_video)
    save_json(stats, stats_file)


if __name__ == '__main__':
    # 1.For test the single_caption function
    # main()

    # 2.Run them in parallel processing
    # test(0, 7500)
    # test(7500, 15000)
    # test(15000, 22500)
    # test(22500, 30000)

    # 3.Merge the video caption jsonl of the caption_cache into the metadata
    # input_file = 'metadata_v3.json'
    # caption_cache_dir = './caption_cache'
    # output_file = 'metadata_v4.json'
    # merge_to_meta(input_file, caption_cache_dir, output_file)

    # 4.Sample and split the metadata for MUVideo and MUImage separately
    metadata = 'metadata_v4.json'
    output_image = 'metadata_image_split.json'
    output_video = 'metadata_video_split.json'
    stats_file = 'split_stats.json'
    sample_and_split(metadata, output_image, output_video, stats_file)