import numpy as np
import os
from PIL import Image

image_size = (64, 64)
images_path = "images/img_align_celeba"

image_tensors = []

for image_name in os.listdir(images_path):
    image = Image.open(os.path.join(images_path, image_name))
    image = image.resize(image_size)
    # image_array = np.array(image) / 255.0
    image_tensors.append(np.array(image))

np.save("C:/Users/tymur.arduch/Desktop/data/celeb/photos_tensor_64_64.npy", np.array(image_tensors, dtype=np.float32))
