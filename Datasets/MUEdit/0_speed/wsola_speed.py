import numpy as np
# import soundfile as sf
from pytsmod import wsola
from pydub import AudioSegment
import json, os
from tqdm import tqdm

# define a function to read an MP3 file into a NumPy array
def mp3_to_numpy(filename):
    audio = AudioSegment.from_mp3(filename)
    audio = audio.set_channels(1)  # turn stereo into mono
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2**15)  # normalization
    fs = audio.frame_rate
    return samples, fs

# define a function to write a NumPy array to an MP3 file
def numpy_to_mp3(samples, fs, filename):
    samples = (samples * (2**15)).astype(np.int16)  # denormalization
    audio = AudioSegment(samples.tobytes(), frame_rate=fs, sample_width=samples.dtype.itemsize, channels=1)
    audio.export(filename, format="mp3")

def wsola_speed(input_file, output_file, speed_factor):
    # load the original MP3 file
    samples, fs = mp3_to_numpy(input_file)
    # adjust the dimensions to adhere to the expected shape
    samples = samples.reshape(1, -1)  # (n_channels, time)
    # wsola algorithm to change the speed of the audio without changing the pitch
    stretched_samples = wsola(samples, speed_factor)
    # save as an MP3 file
    numpy_to_mp3(stretched_samples.flatten(), fs, output_file)
    print(f"Finished, and output file is saved as {output_file}.")

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    file = "../metadata_snr10_0.json"
    path_prefix = '../'
    metadata = load_json(file)
    speed_factors = [0.5, 0.7, 1, 1.3, 1.5]
    speed_name = list(map(lambda x: str(float(x)).replace(".", "_"), speed_factors))
        
    for num, item in enumerate(tqdm(metadata)):
        input_file = os.path.join(path_prefix, item["audio"])
        file_name = os.path.basename(input_file)

        for dir_name, speed_factor in zip(speed_name, speed_factors):
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"Created {dir_name} sub-directory for speed factor of {speed_factor}.")

            output_file = os.path.join(dir_name, file_name)
            if speed_factor == 1:
                # copy the original file)
                os.system(f"cp {input_file} {output_file}")
            else:
                wsola_speed(input_file, output_file, speed_factor)
                
        print(f"Processed {num}-th {file_name} with all speed factors.")

    print("Finished processing all files.")

if __name__ == "__main__":
    main()