import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
else:
    print("No CUDA - trying to build with:", torch.backends.cudnn.version())
