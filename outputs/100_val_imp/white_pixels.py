# 24 July, 2019
# To manually create labels with buildings in mapbox data

from PIL import Image
import numpy as np
import os
import cv2

# for windows
PROJECT_PATH = r"C:\Users\adsampat\Documents\Python_Scripts\My GitHub\unet\unet\100_val_imp"



def load_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(
            os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (256, 256))
        # for y in range(image.shape[0]):
        #     for x in range(image.shape[1]):
        #         if image[y][x] > 220:
        #             image[y][x] = 255
        if image is not None:
            images.append((file, image))
    return images

# path_1 = r"./images"

# images = load_images_from_folder(os.path.join(PROJECT_PATH, path_1))
# for image_file, image in images:
#     print(image_file)
#     image.save(PROJECT_PATH + "/lr/" + image_file, 'png')

path_2 = r"./lr"

print("\nOpenCV")
count_cv = 0
images = load_images(os.path.join(PROJECT_PATH, path_2))
for image_file, image in images:
    count_cv = np.sum(image == 255)
    print("%s: %d" % (image_file, count_cv))
    count_cv = 0


