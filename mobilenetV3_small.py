# I will take MobileNetV3Small as the pretrained model,
# due to the performance of my PC
import keras
from model import X_train, X_val, y_train, y_val

if __name__ == "__main__":
    mobile_net = keras.applications.MobileNetV3Small(weights="imagenet",
                                                     include_top=False)
    # Customize the model
    input_ = keras.layers.Input((128, 128, 3))
    x = keras.applications.mobilenet_v3.preprocess_input(input_)
    x = mobile_net(x)
    x = keras.layers.GlobalAvgPool2D()(x)  # include GlobalAvgPool2D
    x = keras.layers.Dropout(0.5)(x)
    output_ = keras.layers.Dense(40, activation="sigmoid")(x)
    transfer_mobile_net = keras.Model(input_, output_)
    # 1.
    # Freeze the MobileNet layers
    for layer in mobile_net.layers:
        layer.trainable = False
    # Compile the model
    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.8)
    loss_fn = keras.losses.BinaryCrossentropy()
    transfer_mobile_net.compile(optimizer=optimizer, loss=loss_fn)
    # Train the model
    transfer_mobile_net.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,
                            epochs=5)
    # 2.
    # Unfreeze some of the MobileNet layers
    for layer in mobile_net.layers[15:]:
        layer.trainable = True
    # Compile the model
    optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.8)
    loss_fn = keras.losses.BinaryCrossentropy()
    transfer_mobile_net.compile(optimizer=optimizer, loss=loss_fn)
    # Train the model
    transfer_mobile_net.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,
                            epochs=10)

    # Save the model
    transfer_mobile_net.save('models/MobileNetV3_Small.keras')
