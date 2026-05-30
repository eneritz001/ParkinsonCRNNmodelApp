import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_crnn(input_shape):
    
    l2 = regularizers.l2(0.0001)

    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),

        # Convolutional block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),

        # Convolutional block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),

        # Reshape to switch to LSTM 
        layers.Reshape(target_shape=(input_shape[0], -1)),

        # Recurrent block 
        layers.Bidirectional(layers.LSTM(
            64, return_sequences=True,
            kernel_regularizer=l2
        )),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(
            32,
            kernel_regularizer=l2
        )),
        layers.Dropout(0.35),

        # Dense classifier
        layers.Dense(64, activation='relu', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
