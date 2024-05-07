import json, os
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def main():
    file = "../metadata_snr10_1.json"
    metadata = load_json(file)
    prefix = '../'
    # pitch_factors = [-200, -100, 0, 100, 200] # pitch factor in cents
    # pitch_name = ['down_200', 'down_100', 'original', 'up_100', 'up_200']
    pitch_factors = [-500, 500]
    pitch_name = ['down_500', 'up_500']

    for num, item in enumerate(tqdm(metadata)):
        input_file = os.path.join(prefix, item["audio"])
        file_name = os.path.basename(input_file)

        for dir_name, pitch_factor in zip(pitch_name, pitch_factors):
            output_file = os.path.join(dir_name, file_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            
            if pitch_factor == 0:
                output_file = os.path.join(dir_name, file_name)
                os.system(f"cp {input_file} {output_file}")
            else:
                os.system(f"sox {input_file} {output_file} pitch {pitch_factor} norm")
        print(f"Finished {num}-th processing: {file_name}.")
    print("Finished processing all files.")

if __name__ == "__main__":
    main()