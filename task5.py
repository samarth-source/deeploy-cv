from PIL import Image
import numpy as np
import cv2


def check_flag(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to RGB (in case it's not)
    img = img.convert("RGB")

    # Convert image to a numpy array for easier manipulation
    img_array = np.array(img)

    # Get the dimensions of the image
    height, width, _ = img_array.shape

    # Split the image into top and bottom halves
    top_half = img_array[:height // 2, :, :]
    bottom_half = img_array[height // 2:, :, :]

    # Helper function to check if a color is close to red or white
    def is_red_or_white(pixel):
        r, g, b = pixel
        # Red is when R > G and R > B
        is_red = r > g and r > b and r > 100  # Adjust threshold for red
        # White is when R, G, B are close in value
        is_white = abs(r - g) < 30 and abs(r - b) < 30 and r > 200
        return is_red, is_white

    # Check average color in the top and bottom halves
    def average_color(region):
        # Convert to RGB and check each pixel
        red_count = white_count = 0
        total_pixels = region.shape[0] * region.shape[1]

        for row in region:
            for pixel in row:
                is_red, is_white = is_red_or_white(pixel)
                if is_red:
                    red_count += 1
                if is_white:
                    white_count += 1

        # Return the proportion of red and white pixels
        return red_count / total_pixels, white_count / total_pixels

    # Get the average red/white proportions for top and bottom halves
    top_red, top_white = average_color(top_half)
    bottom_red, bottom_white = average_color(bottom_half)

    # Decide based on the proportions
    if top_red > 0.5 and bottom_white > 0.5:  # More red in top, white in bottom
        print("This is the Flag of Indonesia!")
    elif top_white > 0.5 and bottom_red > 0.5:  # More white in top, red in bottom
        print("This is the Flag of Poland!")
    else:
        print("The flag is not recognized as Poland or Indonesia.")


# Example usage
image_path = 'D:\\PyCharm Community Edition 2024.3\\PycharmProjects\\PythonProject\\.venv\\Scripts\\Screenshot 2024-12-17 184557.png' # Replace with the path to your flag image
check_flag(image_path)
