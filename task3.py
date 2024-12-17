import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply a Gaussian blur (low-pass filter)
def apply_low_pass(image, kernel_size=(21, 21)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function to apply high-pass filter (subtract low-pass filtered image from original)
def apply_high_pass(image, low_pass_image):
    return cv2.subtract(image, low_pass_image)

# Function to resize the image to match the target size
def resize_to_match(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

# Input images
image1 = cv2.imread('D:\\PyCharm Community Edition 2024.3\\PycharmProjects\\PythonProject\\.venv\\Scripts\\Screenshot 2024-12-12 231007.png')  # Replace with your first image path
image2 = cv2.imread('D:\\PyCharm Community Edition 2024.3\\PycharmProjects\\PythonProject\\.venv\\Scripts\\Screenshot 2024-12-17 182038.png')  # Replace with your second image path

# Check if images are loaded successfully
if image1 is None or image2 is None:
    print("Error: Could not load one or both images.")
else:
    # Resize image2 to match image1's size (if necessary)
    if image1.shape != image2.shape:
        image2_resized = resize_to_match(image2, image1.shape)
    else:
        image2_resized = image2

    # Step 1: Apply low-pass filter (Gaussian blur) on the second image
    low_pass_image2 = apply_low_pass(image2_resized)

    # Step 2: Apply high-pass filter on the first image
    low_pass_image1 = apply_low_pass(image1)
    high_pass_image1 = apply_high_pass(image1, low_pass_image1)

    # Step 3: Combine high-pass of image1 and low-pass of image2
    combined_image = cv2.add(high_pass_image1, low_pass_image2)

    # Step 4: Display the images
    plt.figure(figsize=(15, 8))

    # Show original image1
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title("Original Image1")
    plt.axis('off')

    # Show original image2
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Original Image2")
    plt.axis('off')

    # Show high-pass image of image1
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(high_pass_image1, cv2.COLOR_BGR2RGB))
    plt.title("High-Pass of Image1")
    plt.axis('off')

    # Show low-pass image of image2
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(low_pass_image2, cv2.COLOR_BGR2RGB))
    plt.title("Low-Pass of Image2")
    plt.axis('off')

    # Show combined image
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title("Combined Image")
    plt.axis('off')

    # Display the images
    plt.tight_layout()
    plt.show()