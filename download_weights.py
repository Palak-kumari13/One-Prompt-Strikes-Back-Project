# import os
# import urllib.request

# # 1. Create the pretrained directory if it doesn't exist
# os.makedirs("pretrained", exist_ok=True)

# # 2. Define the direct resolve URL and output path
# url = "https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/resolve/main/pytorch_model.bin"
# output_path = os.path.join("pretrained", "vit_base_patch16_224_in21k.pth")

# # 3. Download the file
# print("Downloading checkpoint...")
# urllib.request.urlretrieve(url, output_path)
# print(f"Successfully saved to: {output_path}")

import os
import urllib.request

# 1. Create the pretrained directory
os.makedirs("pretrained", exist_ok=True)

# 2. Define URL and path with a .bin extension
url = "https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/resolve/main/pytorch_model.bin"
output_path = os.path.join("pretrained", "pytorch_model.bin")

# 3. Download the file
print("Downloading .bin checkpoint...")
urllib.request.urlretrieve(url, output_path)
print(f"Successfully saved to: {output_path}")