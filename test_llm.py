# import torch
# from transformers import T5Tokenizer
# import time
# # Load the tokenizer and model once at the start
# tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
# model = torch.load("./quantized_lamini_t5.pth")  # Load the quantized model
# model.eval()  # Set the model to evaluation mode
#
# def chatbot_response(input_text):
#     inputs = tokenizer(input_text, return_tensors="pt")
#
#     # Generate response in a single pass
#     with torch.no_grad():
#         output_ids = model.generate(inputs['input_ids'], max_new_tokens=50)
#
#     # Decode the full response at once
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)
#
# def print_typing_effect(text, delay=0.01):  # Reduce delay for faster typing effect
#     for char in text:
#         print(char, end='', flush=True)
#         time.sleep(delay)  # Adjust delay for typing speed
#     print()  # Move to the next line after the response
#
# print("Chatbot: Hello! How can I assist you today?")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit', 'stop']:
#         print("Chatbot: Goodbye!")
#         break
#
#     response = chatbot_response(user_input)
#     print_typing_effect(response)  # Print response with typing effect



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
