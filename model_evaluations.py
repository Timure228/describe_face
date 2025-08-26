import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from model import X_test, y_test, celeba_csv
from model_resnet_34 import ResidualUnit
from PIL import Image
import glob

# Load the model
model = keras.models.load_model("models/model_cnn.keras")
# res_net_34 = keras.models.load_model("models/res_net_34.keras")
# mobilenet = keras.models.load_model("models/MobileNetV3_Small.keras")

# Preprocess the image
img = Image.open("example.jpg")
img_tensor = tf.convert_to_tensor(img) / 255
img_tensor_resized = keras.layers.Resizing(height=128, width=128,
                                           crop_to_aspect_ratio=True)(img_tensor)


def predict_img(model: keras.models.Model, image=None, image_path=None, plot_img=False, as_int=False, custom=False):
    if image and image_path is None:
        return "No image provided"
    # Custom image preprocessing
    if custom and image_path is not None:
        image = Image.open(image_path)
        image_tensor = tf.convert_to_tensor(img) / 255
        image_tensor_resized = keras.layers.Resizing(height=128, width=128,
                                                     crop_to_aspect_ratio=True)(image_tensor)
    # Make a prediction
    y_pred = model.predict(tf.expand_dims(image_tensor_resized, 0))
    # List of the attributes
    attr_list = list(celeba_csv)[1:]
    # Turn into percents
    if as_int:
        return dict(zip(attr_list, y_pred[0]))
    percents = [f"{i:.3}%" for i in y_pred[0] * 100]
    # Plot the image
    if plot_img:
        plt.imshow(image)
        plt.show()

    return dict(zip(attr_list, percents))


def describe(model: keras.models.Model, image) -> None:
    # Get the predictions
    predictions = predict_img(model, image, as_int=True)

    # Hair
    hair_attr = ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair"]
    # Choose the hair type with the highest probability
    prev_hair = "Bald"
    for hair in hair_attr:
        if predictions[hair] > predictions[prev_hair]:
            hair_type = f"{hair} ({100 * predictions[hair]:.1f}%)"
        prev_hair = hair

    # Attractiveness (ugly, average, pretty, beautiful)
    if predictions["Attractive"] <= np.float32(0.25):
        attractiveness = "Ugly"
    elif predictions["Attractive"] <= np.float32(0.6):
        attractiveness = "Average"
    elif predictions["Attractive"] <= np.float32(0.8):
        attractiveness = "Pretty"
    else:
        attractiveness = "Beautiful"

    # Plot the image
    plt.imshow(image)
    plt.axis(False)
    plt.title(f"Hair: {hair_type} \n"
              f"Attractiveness: {attractiveness} ({100 * predictions["Attractive"]:.1f}%)")
    plt.show()


def save_by_attribute(model: keras.models.Model,
                      attribute: str,
                      attr_threshold: float,
                      images_path: str,
                      save_path: str) -> None:
    resizing_layer = keras.layers.Resizing(height=128, width=128, crop_to_aspect_ratio=True)

    for image_path in glob.glob(os.path.join(images_path, "*.jpg")):
        # Preprocess the image
        image = Image.open(image_path)
        img_tensor = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
        img_tensor_resized = resizing_layer(img_tensor)
        # Predict
        y_pred = predict_img(model, img_tensor_resized)
        print(y_pred)
        # Measure
        if float(y_pred[attribute][:-1]) > attr_threshold:
            # Save the image
            img_file_name = os.path.basename(image_path)
            new_img_dir = os.path.join(save_path, f"{attribute}_{img_file_name}")
            shutil.copyfile(image_path, new_img_dir)


def printout_prf(model, labels, samples):
    # Get the predictions
    y_test_pred = model.predict(samples)

    # Define the Recall
    recall = keras.metrics.Recall()
    recall.update_state(labels, y_test_pred)

    recall_score = recall.result()
    # Define the Precision
    precision = keras.metrics.Precision()
    precision.update_state(labels, y_test_pred)

    precision_score = precision.result()
    # Calculate the f1 score
    f1_score = ((2 * precision.result() * recall.result()) /
                (precision.result() + recall.result()))

    return (f"Recall: {100 * recall_score :.1f}% \n"
            f"Precision: {100 * precision_score :.1f}% \n"
            f"F1-Score: {100 * f1_score :.1f}%")


# print(predict_img(mobilenet, X_test[343], True))

# Regular model
# print(printout_prf(model=model, labels=y_test, samples=X_test))
# epochs=200, momentum=0.1, learning_rate=0.001, model=model
# Recall: 55.6%
# Precision 79.3%
# F1-Score 65.4%

# ResNet-34
# print(printout_prf(model=res_net_34, labels=y_test, samples=X_test))
# epochs=1, momentum=0.1, learning_rate=0.001, model=res_net_34
# Recall: 32.7%
# Precision 60.5%
# F1-Score 42.5%

# MobileNetV3_Small
# print(printout_prf(model=mobilenet, labels=y_test, samples=X_test))
# epochs=[5, 10], momentum=0.8, learning_rate=[0.1, 0.001], model=mobilenet
# Recall: 39.4%
# Precision 51.5%
# F1-Score 44.6%

# save_by_attribute(model=mobilenet, attribute="Attractive", attr_threshold=50.0,
#                   images_path="images/test_images",
#                   save_path="images/blond_people")

print(predict_img(model, custom=True, image_path="example.jpg", plot_img=True))
# print(describe(model, img_tensor_resized))
