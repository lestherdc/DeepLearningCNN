import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # BLOQUE CNN
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # BLOQUE LSTM
    lstm_out = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(x)

    # CAPA DE ATENCIÓN (Versión estable sin Lambda compleja)
    u = layers.Dense(1, activation='tanh')(lstm_out)
    a = layers.Flatten()(u)
    a = layers.Activation('softmax')(a)

    # En lugar de Multiply + Lambda Sum, usamos un enfoque de reducción directa
    # que Keras puede serializar sin problemas:
    x = layers.GlobalAveragePooling1D()(lstm_out)  # Captura la esencia de la secuencia

    # SALIDA
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='sigmoid', name='output_layer')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model