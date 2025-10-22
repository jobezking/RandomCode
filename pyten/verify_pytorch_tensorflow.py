import torch
import tensorflow as tf

print("--- PyTorch Verification ---")
# Check if PyTorch can access CUDA
if torch.cuda.is_available():
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"PyTorch found {gpu_count} CUDA-enabled GPU(s).")

    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("PyTorch cannot find a CUDA-enabled GPU. 😥")

print("\n--- TensorFlow Verification ---")
# List physical devices (CPUs and GPUs)
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print(f"TensorFlow found {len(gpus)} GPU(s).")
    # Loop through and print details (optional)
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
else:
    print("TensorFlow cannot find a GPU. 😥")
