from safetensors.torch import load_file
import numpy as np

def compare_safetensors(file1, file2):
    # Load the safetensors files
    weights1 = load_file(file1)
    weights2 = load_file(file2)

    # Get the keys (weight names) from both files
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())

    # Check if the keys are the same
    if keys1 != keys2:
        print("The weight names (keys) are different!")
        print("Keys in file1 but not in file2:", keys1 - keys2)
        print("Keys in file2 but not in file1:", keys2 - keys1)

    print("The weight names (keys) are the same.")

    # Check if the values for each key are the same
    all_match = True
    for key in keys1:
        if key in keys2:
            value1 = weights1[key].numpy()
            value2 = weights2[key].numpy()

            if not np.array_equal(value1, value2):
                print(f"Values for key '{key}' are different!")
                all_match = False

    if all_match:
        print("All weights have the same values.")
    else:
        print("Some weights have different values.")

# Example usage
if __name__ == "__main__":
    file1 = "/opt/nvme0/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/video_encoder_base.safetensors"
    file2 = "/opt/nvme0/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/ckpts/video_encoder_base_new.safetensors"
    compare_safetensors(file1, file2)