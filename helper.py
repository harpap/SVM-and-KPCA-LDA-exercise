import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current device index:", torch.cuda.current_device())
    print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("No GPU available to PyTorch.")
