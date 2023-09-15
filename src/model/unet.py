import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.python.keras.models import Model


def unet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Mais camadas do encoder...

    # Decoder
    up6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', activation='relu')(pool5)
    merge6 = concatenate([conv1, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)

    # Mais camadas do decoder...

    # Camada de saída
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv10)

    model = Model(inputs, outputs)

    return model
