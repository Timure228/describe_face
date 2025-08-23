import tensorflow as tf
import numpy as np
import pandas as pd
import keras

# CSV (labels)
celeba_csv = pd.read_csv("data/celeba.csv")

csv_no_id = celeba_csv.drop("ID", axis=1)

csv_copy = csv_no_id.copy()

# NPY (photo tensors)
celeba_photos = np.load("data/photos_tensor.npy")

celeba_tensors = tf.convert_to_tensor(celeba_photos)

# Set train, test and validation splits
train_split, val_split, test_split = int(len(celeba_tensors) * 0.7), int(len(celeba_tensors) * 0.2), int(
    len(celeba_tensors) * 0.1)

X_train, y_train = celeba_tensors[:train_split], csv_copy[:train_split]
X_val, y_val = celeba_tensors[train_split:train_split + val_split], csv_copy[train_split:train_split + val_split]
X_test, y_test = celeba_tensors[train_split + val_split:], csv_copy[train_split + val_split:]

# The model
import tensorflow as tf

# Define Leaky ReLu activation function
leaky_relu = tf.nn.leaky_relu
# Define the model architecture
model_cnn = keras.Sequential([
    keras.layers.Input((128, 128, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=11, strides=4, activation=leaky_relu),
    keras.layers.MaxPooling2D(strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation=leaky_relu),
    keras.layers.MaxPooling2D(strides=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=leaky_relu),
    keras.layers.MaxPooling2D(strides=2),

    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation=leaky_relu, kernel_initializer="he_normal"),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(40, activation=keras.activations.sigmoid)
])

# Define optimizer and loss function
optimizer = keras.optimizers.SGD(momentum=0.1, learning_rate=0.001)
loss = keras.losses.BinaryCrossentropy()

# Train
if __name__ == "__main__":
    # Compile the model
    model_cnn.compile(optimizer=optimizer, loss=loss,
                      metrics=["accuracy"])

    # Define the callbacks
    early_stop = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
    # Train the model
    history = model_cnn.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,
                            epochs=200, callbacks=[early_stop])
    # Show the model architecture
    model_cnn.summary()
    # Save the model
    model_cnn.save('models/model_cnn.keras')
