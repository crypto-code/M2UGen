import os

def convert_videos(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # iterate through all files in the input folder
    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        
        for file in files:
            if file.endswith(".mp4"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)
                print(output_path)
                # scale the video to 204x360, h264 codec
                ffmpeg_command = f'ffmpeg -i "{input_path}" -vf scale=204:360 -c:v libx264 "{output_path}"'
                os.system(ffmpeg_command)
                print(f"Converted {input_path} to {output_path}")

input_folder = "audioset_videos"
output_folder = "audioset_videos_scaled_full"

convert_videos(input_folder, output_folder)
