'''import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pretrained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("MBZUAI/LaMini-T5-738M")
tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")

# Set the model to evaluation mode
model.eval()

# Prepare the model for quantization
model = torch.quantization.quantize_dynamic(
    model,  # the model to quantize
    {torch.nn.Linear},  # layers to quantize
    dtype=torch.qint8  # quantization type
)

torch.save(model, "./quantized_lamini_t5.pth")
print("model saved successfully")


import os

# Path to the quantized model file
model_path = "./quantized_lamini_t5.pth"

# Get the size of the model file
model_size = os.path.getsize(model_path)  # Size in bytes

# Convert size to megabytes for better readability
model_size_mb = model_size / (1024 * 1024)  # Convert bytes to MB

print(f"Quantized Model Size: {model_size_mb:.2f} MB")
'''

#
# #quantisaton code
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pretrained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("MBZUAI/LaMini-T5-738M")
tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")

# Set the model to evaluation mode
model.eval()

# Prepare the model for quantization
model = torch.quantization.quantize_dynamic(
    model,  # the model to quantize
    {torch.nn.Linear},  # layers to quantize
    dtype=torch.qint8  # quantization type
)

torch.save(model, "./quantized_lamini_t5.pth")
print("model saved successfully")


import os

# Path to the quantized model file
model_path = "./quantized_lamini_t5.pth"

# Get the size of the model file
model_size = os.path.getsize(model_path)  # Size in bytes

# Convert size to megabytes for better readability
model_size_mb = model_size / (1024 * 1024)  # Convert bytes to MB

print(f"Quantized Model Size: {model_size_mb:.2f} MB")
