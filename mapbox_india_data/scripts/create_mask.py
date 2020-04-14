"""
24 July, 2019
Pranav Bijapur

To create label masks for manually labelled tiles with buildings in mapbox data
"""

from PIL import Image
import os

# for windows
PROJECT_PATH = "E:\\projects\\Satellite Image Processing\\satellite-image-segmentation\\mapbox_india_data\\"


def load_images_from_folder(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = Image.open(os.path.join(folder_path, file))
        if image is not None:
            images.append((file, image))
    return images


images = load_images_from_folder(PROJECT_PATH + "data_first_150/annotated_tiles/")
for image_file, image in images:
    pixels = image.load()
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            # use (255, 0, 255) in Paint 3D
            if pixels[x, y][0] > 220 and pixels[x, y][1] < 20 and pixels[x, y][2] > 220:
                pixels[x, y] = (255, 255, 255)
            else:
                pixels[x, y] = (0, 0, 0)
    print("writing " + image_file)
    image.save(PROJECT_PATH + "data_first_150/labels/" + image_file)
