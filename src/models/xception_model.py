import tensorflow as tf
from keras.applications import Xception

def create_xception_model(num_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Adicionar a camada de saída para classificação
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model
