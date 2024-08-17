from PIL import Image
from tqdm import tqdm
import os

def resize_images(input_folder, output_folder, target_size=(1024, 1024)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subfolder_count = sum([len(dirs) for _, dirs, _ in os.walk(input_folder)])

    for root, dirs, files in tqdm(os.walk(input_folder), total=subfolder_count, desc="Processing folders", unit="folder"):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                with Image.open(input_path) as img:
                    resized_img = img.resize(target_size)

                    # Save the resized image
                    resized_img.save(output_path)

if __name__ == "__main__":
    input_folder = "path"
    output_folder = "path"
    target_size = (1024, 1024)

    resize_images(input_folder, output_folder, target_size)
