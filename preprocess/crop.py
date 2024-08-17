import cv2
import os
from tqdm import tqdm

def crop_black_space(input_folder, output_folder, padding):
    subfolder_count = sum([len(dirs) for _, dirs, _ in os.walk(input_folder)])
    for root, _, files in tqdm(os.walk(input_folder), total=subfolder_count, desc="Processing folders", unit="folder"):
        rel_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, rel_path)

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for file in files:
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                crop_black_space_single(input_path, output_path, padding)

def crop_black_space_single(image_path, output_path, padding=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1], w + 2 * padding)
    h = min(img.shape[0], h + 2 * padding)
    cropped_img = img[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_img)

if __name__ == "__main__":
    input_folder = "path"
    output_folder = "path"
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    crop_black_space(input_folder, output_folder, padding=15)

