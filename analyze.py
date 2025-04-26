import cv2
import numpy as np

# Use absolute paths for your images
image1_path = "E:/COLLEGE_PROJECTS/vehicle_detection_gemini/vehicle_image.jpg"
image2_path = "E:/COLLEGE_PROJECTS/vehicle_detection_gemini/vehicle_image1.jpeg"

def compare_vehicles(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Check if images are loaded properly
    if image1 is None:
        print(f"Error: Unable to load image: {image1_path}")
        return
    if image2 is None:
        print(f"Error: Unable to load image: {image2_path}")
        return
    
    # Convert images to grayscale (ORB works on grayscale images)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches based on distance (lower distance means better match)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw the matches on the images for visualization
    result_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Calculate the percentage of good matches (threshold could be adjusted)
    good_match_threshold = 0.3  # percentage threshold for good matches
    good_matches = [m for m in matches if m.distance < 50]  # Distance threshold for "good" matches
    
    # Show the result
    cv2.imshow('Keypoint Matches', result_image)
    
    # Determine similarity based on the number of good matches
    match_ratio = len(good_matches) / len(matches)
    print(f"Match ratio: {match_ratio * 100:.2f}%")
    
    if match_ratio > good_match_threshold:
        print("The vehicles are likely the same!")
    else:
        print("The vehicles are different!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

compare_vehicles(image1_path, image2_path)
