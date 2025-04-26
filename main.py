import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import pathlib
from datetime import datetime
import glob

# ========== SETTINGS ==========
api_key = "YOUR_API_KEY"  # Your Gemini API Key
camera_mode = False  # Set to False to load from folder, True to use webcam
image_folder = r"E:\COLLEGE_PROJECTS\vehicle_detection_gemini"  # Folder where test images are stored (for OFF mode)
excel_file = "vehicle_data.xlsx"  # Excel file to store the data
previous_image_path = None  # Keeps track of the previous vehicle
# ===============================

# Configure Gemini API
genai.configure(api_key=api_key)
orb = cv2.ORB_create()

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # 0 = default camera
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"captured_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        cap.release()
        return image_path
    else:
        cap.release()
        print("Failed to capture image.")
        return None

# Function to load images from folder (sorted by name)
def load_images_from_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    image_paths += glob.glob(os.path.join(folder_path, "*.jpeg"))
    image_paths += glob.glob(os.path.join(folder_path, "*.png"))
    image_paths.sort()
    return image_paths

# Function to analyze vehicle image using Gemini
def analyze_vehicle_with_gemini(image_path):
    img = Image.open(image_path)

    prompt = """
    You are analyzing a vehicle image. Please extract the following:
    1. Vehicle number plate text (if visible)
    2. Vehicle name or brand seen on the glass or body
    3. Any suspicious signs that the number plate may be fake or replaced
    """

    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    response = model.generate_content([prompt, img])

    return response.text

# Function to compare two vehicle images using ORB
def compare_vehicles_with_orb(new_image_path, previous_image_path):
    image1 = cv2.imread(new_image_path)
    image2 = cv2.imread(previous_image_path)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_ratio = len(matches) / max(len(kp1), len(kp2))
    return match_ratio, matches

# Function to save results into Excel
def store_vehicle_data_in_excel(vehicle_data):
    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=["Timestamp", "Vehicle Number Plate", "Vehicle Brand", "Match Ratio"])
        df.to_excel(excel_file, index=False)

    df = pd.read_excel(excel_file)
    df = pd.concat([df, pd.DataFrame([vehicle_data])], ignore_index=True)
    df.to_excel(excel_file, index=False)

# Extract number plate and brand info from Gemini text
def extract_info_from_gemini_response(response_text):
    number_plate = "Not Found"
    brand = "Not Found"

    lines = response_text.split('\n')
    for line in lines:
        if "Vehicle number plate" in line:
            number_plate = line.split(":")[-1].strip()
        if "Vehicle name" in line or "brand" in line:
            brand = line.split(":")[-1].strip()

    return number_plate, brand

# Main function
def main():
    global previous_image_path

    if camera_mode:
        while True:
            print("\n[INFO] Capturing vehicle image...")
            new_image_path = capture_image()
            if not new_image_path:
                continue

            response_text = analyze_vehicle_with_gemini(new_image_path)
            print("\n🔍 Gemini's Vehicle Analysis Result:")
            print(response_text)

            number_plate, brand = extract_info_from_gemini_response(response_text)

            match_ratio = 0
            if previous_image_path:
                match_ratio, matches = compare_vehicles_with_orb(new_image_path, previous_image_path)
                print(f"Match Ratio: {match_ratio * 100:.2f}%")

            store_vehicle_data_in_excel({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Vehicle Number Plate": number_plate,
                "Vehicle Brand": brand,
                "Match Ratio": match_ratio
            })

            previous_image_path = new_image_path
    else:
        image_paths = load_images_from_folder(image_folder)
        if not image_paths:
            print(f"No images found in {image_folder}")
            return

        for new_image_path in image_paths:
            print(f"\n[INFO] Processing {new_image_path}...")

            response_text = analyze_vehicle_with_gemini(new_image_path)
            print("\n🔍 Gemini's Vehicle Analysis Result:")
            print(response_text)

            number_plate, brand = extract_info_from_gemini_response(response_text)

            match_ratio = 0
            if previous_image_path:
                match_ratio, matches = compare_vehicles_with_orb(new_image_path, previous_image_path)
                print(f"Match Ratio: {match_ratio * 100:.2f}%")

            store_vehicle_data_in_excel({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Vehicle Number Plate": number_plate,
                "Vehicle Brand": brand,
                "Match Ratio": match_ratio
            })

            previous_image_path = new_image_path

if __name__ == "__main__":
    main()
