from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to detect flag type (Indonesia or Poland) using dominant color analysis
def detect_flag(image):
    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into top and bottom halves
    height, width, _ = image.shape
    top_half = image[:height // 2, :]
    bottom_half = image[height // 2:, :]

    # Function to classify color based on RGB values
    def classify_color(pixel):
        # Red color detection (simple threshold based on RGB channels)
        if pixel[0] > 100 and pixel[1] < 80 and pixel[2] < 80:  # Red
            return "Red"
        # White color detection (based on high values in all RGB channels)
        elif pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200:  # White
            return "White"
        else:
            return "Unknown"

    # Function to count red and white pixels in a region
    def count_red_white_pixels(half_image):
        red_pixels = 0
        white_pixels = 0
        for row in half_image:
            for pixel in row:
                color = classify_color(pixel)
                if color == "Red":
                    red_pixels += 1
                elif color == "White":
                    white_pixels += 1
        return red_pixels, white_pixels

    # Count pixels in both top and bottom halves
    top_red, top_white = count_red_white_pixels(top_half)
    bottom_red, bottom_white = count_red_white_pixels(bottom_half)

    # Determine which flag it is based on the pixel count
    if top_red > top_white and bottom_white > bottom_red:
        return "Indonesia Flag"
    elif top_white > top_red and bottom_red > bottom_white:
        return "Poland Flag"
    else:
        return "Unkown Flag"


# Function to handle cases with only flag
def handle_only_flag(image):
    # Directly apply the flag detection function without relying on YOLO
    flag_type = detect_flag(image)
    print(f"Detected Flag: {flag_type}")

    # Show the image without bounding box
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Flag: {flag_type}")
    plt.axis("off")
    plt.show()


# Load the image
image_path = r"D:\PyCharm Community Edition 2024.3\PycharmProjects\PythonProject\.venv\Scripts\Screenshot 2024-12-17 184602.png"
image = cv2.imread(image_path)
# Perform object detection
model = YOLO("yolov8n.pt")  # Replace with your YOLO model
results = model(image)

# Check if the image contains any objects detected
flag_bbox = None
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        label = result.names[class_id]
        if label.lower() == "kite":  # Check if the detected object is a kite (flag is detected as kite)
            # Manually change label to "flag"
            result.names[class_id] = "flag"
            flag_bbox = box.xyxy  # Get the bounding box coordinates (x1, y1, x2, y2)
            break
    if flag_bbox is not None:
        break

# If no flag is detected by YOLO, assume the image contains only the flag
if flag_bbox is None:
    print("No objects detected by YOLO. Assuming image contains only the flag.")
    handle_only_flag(image)
else:
    # If a flag is detected via YOLO, crop and classify it
    print("Flag detected using object detection.")
    # Crop the detected flag region with some padding to ensure it covers the entire flag
    x1, y1, x2, y2 = map(int, flag_bbox[0])  # Get the coordinates of the bounding box
    padding = 10  # Padding to crop a bit outside the bounding box, just in case
    cropped_flag = image[max(0, y1 - padding):y2 + padding, max(0, x1 - padding):x2 + padding]

    # Determine the flag type (Indonesia or Poland)
    flag_type = detect_flag(cropped_flag)
    print(flag_type)

    # Draw a rectangle around the detected flag
    image_with_box = image.copy()  # Create a copy of the original image to draw the box on
    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

    # Show the image with the bounding box
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Flag: {flag_type}")
    plt.axis("off")
    plt.show()
