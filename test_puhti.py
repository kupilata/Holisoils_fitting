#source /scratch/project_2010938/venv_mine/bin/activate

# import the libraries
import torch
import time

# Check CUDA availability
cuda_available = torch.cuda.is_available()
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "No CUDA device"

# Print to console
print("CUDA available:", cuda_available)
print("CUDA device name:", cuda_device_name)
print(torch.version.cuda)          # e.g., '11.7'
print(torch.backends.cudnn.enabled)  # Should be True

# Save to file
with open("cuda_status.txt", "w") as f:
    f.write(f"CUDA available: {cuda_available}\n")
    f.write(f"CUDA device name: {cuda_device_name}\n")