import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_crnn(input_shape):
    # L2 reducido a 0.0001 (era 0.001).
    # Con 7 capas acumulando penalizacion, 0.001 era demasiado agresivo:
    # loss arrancaba en ~1.5 en lugar de ~0.4, señal de que L2 dominaba
    # sobre la binary_crossentropy y el modelo no podia aprender bien.
    # Solo se aplica a Dense y LSTM (las capas mas propensas a overfitting),
    # no a Conv2D (los filtros convolucionales son pequeños y raramente se disparan).
    l2 = regularizers.l2(0.0001)

    model = models.Sequential([
        # --- Entrada ---
        layers.Input(shape=input_shape),

        # --- Bloque convolucional 1 (sin L2 — filtros pequeños, bajo riesgo) ---
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),

        # --- Bloque convolucional 2 ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),

        # --- Reshape para pasar a LSTM ---
        layers.Reshape(target_shape=(input_shape[0], -1)),

        # --- Bloque recurrente (L2 solo en kernel, no en recurrent) ---
        # recurrent_regularizer tambien se elimina: con dataset augmentado
        # es suficiente con kernel_regularizer + Dropout para las LSTM.
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

        # --- Clasificador denso (L2 aqui si tiene mas sentido) ---
        layers.Dense(64, activation='relu', kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
