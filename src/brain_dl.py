import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # BLOQUE CNN (Reconocimiento de patrones visuales en la serie)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # BLOQUE LSTM (Memoria de largo plazo sobre niveles)
    lstm_out = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(x)

    # CAPA DE ATENCIÓN (Enfoque en eventos clave)
    u = layers.Dense(1, activation='tanh')(lstm_out)
    a = layers.Flatten()(u)
    a = layers.Activation('softmax')(a)
    a = layers.RepeatVector(200)(a)  # 200 por Bidirectional(100)
    a = layers.Permute([2, 1])(a)

    combined = layers.Multiply()([lstm_out, a])
    context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(combined)

    # SALIDA
    x = layers.Dense(64, activation='relu')(context)
    x = layers.Dropout(0.3)(x)
    # Sigmoid es mejor para clasificación multietiqueta (puede tocar ambos niveles)
    outputs = layers.Dense(2, activation='sigmoid', name='output_layer')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model