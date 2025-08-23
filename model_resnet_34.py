import tensorflow as tf
import keras
from model import X_train, y_train, X_val, y_val, loss, optimizer
from functools import partial

# Create the ResidualUnit class
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)


@keras.saving.register_keras_serializable()
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        self.main_layers = [
            DefaultConv2D(self.filters, strides=self.strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(self.filters),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if self.strides > 1:
            self.skip_layers = [
                DefaultConv2D(self.filters, kernel_size=1, strides=self.strides)
            ]

        super().build(input_shape)

    def call(self, inputs):
        X = inputs
        for layer in self.main_layers:
            X = layer(X)
        skip_X = inputs
        for layer in self.skip_layers:
            skip_X = layer(skip_X)

        return self.activation(X + skip_X)


if __name__ == "__main__":
    # Define the ResNet-34 model
    res_net_34 = keras.Sequential([
        keras.layers.Input((128, 128, 3)),
        DefaultConv2D(64, kernel_size=7, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ])
    prev_filter = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filter else 2
        res_net_34.add(ResidualUnit(filters, strides=strides))
        prev_filter = filters

    res_net_34.add(keras.layers.GlobalAvgPool2D())
    res_net_34.add(keras.layers.Flatten())
    res_net_34.add(keras.layers.Dense(40, activation=tf.keras.activations.sigmoid))

    # Compile the model
    res_net_34.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Define Early Stopping
    early_stop = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)

    # Show the model architecture
    res_net_34.summary()

    # Fit the model
    res_net_34.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,
                   epochs=1, callbacks=[early_stop])

    # Save the model
    res_net_34.save('models/res_net_34.keras')
