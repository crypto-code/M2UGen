import torch.cuda

import gradio as gr
import mdtex2html
import tempfile
from PIL import Image
import scipy
import argparse

from llama.mumu_llama import MuMu_LLaMA
import llama
import numpy as np
import os
import torch
import torchaudio
import torchvision.transforms as transforms
import av
import subprocess
import librosa

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="./ckpts/checkpoint.pth", type=str,
    help="Name of or path to MuMu_LLaMA pretrained checkpoint",
)
parser.add_argument(
    "--llama_type", default="7B", type=str,
    help="Type of llama original weight",
)
parser.add_argument(
    "--llama_dir", default="./ckpts/LLaMA-2", type=str,
    help="Path to LLaMA pretrained checkpoint",
)
parser.add_argument(
    "--mert_path", default="m-a-p/MERT-v1-330M", type=str,
    help="Path to MERT pretrained checkpoint",
)
parser.add_argument(
    "--vit_path", default="google/vit-base-patch16-224", type=str,
    help="Path to ViT pretrained checkpoint",
)
parser.add_argument(
    "--vivit_path", default="google/vivit-b-16x2-kinetics400", type=str,
    help="Path to ViViT pretrained checkpoint",
)
parser.add_argument(
    "--knn_dir", default="./ckpts", type=str,
    help="Path to directory with KNN Index",
)
parser.add_argument(
    '--music_decoder', default="musicgen", type=str,
    help='Decoder to use musicgen/audioldm2')

parser.add_argument(
    '--music_decoder_path', default="facebook/musicgen-small", type=str,
    help='Path to decoder to use musicgen/audioldm2')

args = parser.parse_args()

generated_audio_files = []

llama_type = args.llama_type
llama_ckpt_dir = os.path.join(args.llama_dir, llama_type)
llama_tokenzier_path = args.llama_dir
model = MuMu_LLaMA(llama_ckpt_dir, llama_tokenzier_path, args, knn=False, stage=3, load_llama=False)

print("Loading Model Checkpoint")
checkpoint = torch.load(args.model, map_location='cpu')

new_ckpt = {}
for key, value in checkpoint['model'].items():
    key = key.replace("module.", "")
    new_ckpt[key] = value

load_result = model.load_state_dict(new_ckpt, strict=True)
assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
model.eval()
model.to("cuda")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)])


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text, image_path, video_path, audio_path):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    outputs = text
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines) + "<br>"
    if image_path is not None:
        text += f'<img src="./file={image_path}" style="display: inline-block;"><br>'
        outputs = f'<Image>{image_path}</Image> ' + outputs
    if video_path is not None:
        text += f' <video controls playsinline height="320" width="240" style="display: inline-block;"  src="./file={video_path}"></video6><br>'
        outputs = f'<Video>{video_path}</Video> ' + outputs
    if audio_path is not None:
        text += f'<audio controls playsinline><source src="./file={audio_path}" type="audio/wav"></audio><br>'
        outputs = f'<Audio>{audio_path}</Audio> ' + outputs
    # text = text[::-1].replace(">rb<", "", 1)[::-1]
    text = text[:-len("<br>")].rstrip() if text.endswith("<br>") else text
    return text, outputs


def save_audio_to_local(audio, sec):
    global generated_audio_files
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.wav')
    if args.music_decoder == "audioldm2":
        scipy.io.wavfile.write(filename, rate=16000, data=audio[0])
    else:
        scipy.io.wavfile.write(filename, rate=model.generation_model.config.audio_encoder.sampling_rate, data=audio)
    generated_audio_files.append(filename)
    return filename


def parse_reponse(model_outputs, audio_length_in_s):
    response = ''
    text_outputs = []
    for output_i, p in enumerate(model_outputs):
        if isinstance(p, str):
            response += p.replace(' '.join([f'[AUD{i}]' for i in range(8)]), '')
            response += '<br>'
            text_outputs.append(p.replace(' '.join([f'[AUD{i}]' for i in range(8)]), ''))
        elif 'aud' in p.keys():
            _temp_output = ''
            for idx, m in enumerate(p['aud']):
                if isinstance(m, str):
                    response += m.replace(' '.join([f'[AUD{i}]' for i in range(8)]), '')
                    response += '<br>'
                    _temp_output += m.replace(' '.join([f'[AUD{i}]' for i in range(8)]), '')
                else:
                    filename = save_audio_to_local(m, audio_length_in_s)
                    print(filename)
                    _temp_output = f'<Audio>{filename}</Audio> ' + _temp_output
                    response += f'<audio controls playsinline><source src="./file={filename}" type="audio/wav"></audio>'
            text_outputs.append(_temp_output)
        else:
            pass
    response = response[:-len("<br>")].rstrip() if response.endswith("<br>") else response
    return response, text_outputs


def reset_user_input():
    return gr.update(value='')


def reset_dialog():
    return [], []


def reset_state():
    global generated_audio_files
    generated_audio_files = []
    return None, None, None, None, [], [], []


def upload_image(conversation, chat_history, image_input):
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    input_image.save(image_input.name)  # Overwrite with smaller image.
    conversation += [(f'<img src="./file={image_input.name}" style="display: inline-block;">', "")]
    return conversation, chat_history + [input_image, ""]


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        frames.append(frame)
    chosen_frames = []
    for i in indices:
        chosen_frames.append(frames[i])
    return np.stack([x.to_ndarray(format="rgb24") for x in chosen_frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len > seg_len:
        converted_len = 0
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_video_length(filename):
    print("Getting Video Length")
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return int(round(float(result.stdout)))


def get_audio_length(filename):
    return int(round(librosa.get_duration(path=filename)))


def predict(
        prompt_input,
        image_path,
        audio_path,
        video_path,
        chatbot,
        top_p,
        temperature,
        history,
        modality_cache,
        audio_length_in_s):
    global generated_audio_files
    prompts = [llama.format_prompt(prompt_input)]
    prompts = [model.tokenizer(x).input_ids for x in prompts]
    image, audio, video = None, None, None
    if image_path is not None:
        image = transform(Image.open(image_path))
    if audio_path is not None:
        sample_rate = 24000
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        audio = torch.mean(waveform, 0)
    if video_path is not None:
        print("Opening Video")
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

    if len(generated_audio_files) != 0:
        sample_rate = 24000
        waveform, sr = torchaudio.load(generated_audio_files[-1])
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        audio = torch.mean(waveform, 0)

    print(image, video, audio)
    response = model.generate(prompts, audio, image, video, 512, temperature, top_p,
                              audio_length_in_s=audio_length_in_s)
    print(response)
    response_chat, response_outputs = parse_reponse(response, audio_length_in_s)
    print('text_outputs: ', response_outputs)
    user_chat, user_outputs = parse_text(prompt_input, image_path, video_path, audio_path)
    chatbot.append((user_chat, response_chat))
    history.append((user_outputs, ''.join(response_outputs).replace('\n###', '')))
    return chatbot, history, modality_cache, None, None, None,


with gr.Blocks() as demo:
    gr.HTML("""
        <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; "><img src='./file=bot.png' width="50" height="50" style="margin-right: 10px;">M<sup style="line-height: 200%; font-size: 60%">2</sup>UGen</h1>
        <h3>This is the demo page of M<sup>2</sup>UGen, a Multimodal LLM capable of Music Understanding and Generation!</h3>
        <div style="display: flex;"><a href='https://arxiv.org/pdf/2311.11255.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
        """)

    with gr.Row():
        with gr.Column(scale=0.7, min_width=500):
            with gr.Row():
                chatbot = gr.Chatbot(label='MuMu_LLaMA Chatbot', avatar_images=(
                (os.path.join(os.path.dirname(__file__), 'user.png')),
                (os.path.join(os.path.dirname(__file__), "bot.png")))).style(height=440)

            with gr.Tab("User Input"):
                with gr.Row(scale=3):
                    user_input = gr.Textbox(label="Text", placeholder="Key in something here...", lines=3)
                with gr.Row(scale=3):
                    with gr.Column(scale=1):
                        # image_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])
                        image_path = gr.Image(type="filepath",
                                              label="Image")  # .style(height=200)  # <PIL.Image.Image image mode=RGB size=512x512 at 0x7F6E06738D90>
                    with gr.Column(scale=1):
                        audio_path = gr.Audio(type='filepath')  # .style(height=200)
                    with gr.Column(scale=1):
                        video_path = gr.Video()  # .style(height=200) # , value=None, interactive=True
        with gr.Column(scale=0.3, min_width=300):
            with gr.Group():
                with gr.Accordion('Text Advanced Options', open=True):
                    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0, 1, value=0.6, step=0.01, label="Temperature", interactive=True)
                with gr.Accordion('Audio Advanced Options', open=False):
                    audio_length_in_s = gr.Slider(5, 30, value=30, step=1, label="The audio length in seconds",
                                                  interactive=True)
            with gr.Tab("Operation"):
                with gr.Row(scale=1):
                    submitBtn = gr.Button(value="Submit & Run", variant="primary")
                with gr.Row(scale=1):
                    emptyBtn = gr.Button("Clear History")

    history = gr.State([])
    modality_cache = gr.State([])

    submitBtn.click(
        predict, [
            user_input,
            image_path,
            audio_path,
            video_path,
            chatbot,
            top_p,
            temperature,
            history,
            modality_cache,
            audio_length_in_s
        ], [
            chatbot,
            history,
            modality_cache,
            image_path,
            audio_path,
            video_path
        ],
        show_progress=True
    )

    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[
        image_path,
        audio_path,
        video_path,
        chatbot,
        history,
        modality_cache
    ], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=24000)
