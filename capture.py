import google.generativeai as genai
from PIL import Image
import os
import pathlib

# Configure the API key
api_key = "YOUR_API_KEY"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Load the image
image_path = "vehicle_image.jpg"
img = Image.open(image_path)

# Prepare the prompt for analysis
prompt = """
You are analyzing a vehicle image. Please extract the following:
1. Vehicle number plate text (if visible)
2. Vehicle name or brand seen on the glass or body
3. Any suspicious signs that the number plate may be fake or replaced
"""

# For version 0.8.5, use this approach:
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Initialize the model
model = genai.GenerativeModel(model_name="gemini-1.5-flash", 
                             generation_config=generation_config)

# Generate content with the image
response = model.generate_content([prompt, img])

# Print the results
print("\n🔍 Gemini's Vehicle Analysis Result:")
print(response.text)