import numpy as np
from PIL import Image
import os


image_size = (128, 128)
images_path = "images/img_align_celeba"

image_tensors = []
for image_name in os.listdir(images_path):
    image = Image.open(os.path.join(images_path, image_name))
    image = image.resize(image_size)
    image_array = np.array(image) / 255.0
    image_tensors.append(image_array)

np.save("/data/photos_tensor.npy", np.array(image_tensors, dtype=np.float32))
