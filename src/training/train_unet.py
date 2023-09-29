import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.keras.metrics import MeanIoU
from tensorflow.python.keras.optimizer_v1 import Adam
from keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append('/Users/feliz/OneDrive/Documentos/UESPI/8_bloco/tcc_2/codigo')
from src.models.unet import unet

# Crie geradores de dados para treinamento e validação com aumento de dados
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalização das imagens
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

validation_data_generator = ImageDataGenerator(
    rescale=1.0 / 255  # Normalização das imagens de validação
)

train_generator = train_data_generator.flow_from_directory(
    './data/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # Para imagens de entrada
    classes=['1', '2', '3']  # Adicione as classes correspondentes às pastas
)

validation_generator = validation_data_generator.flow_from_directory(
    './data/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # Para imagens de entrada
    classes=['1', '2', '3']  # Adicione as classes correspondentes às pastas
)

batch_size = train_generator.batch_size
num_samples = len(train_generator.filenames)
steps_per_epoch = num_samples // batch_size

batch_size = validation_generator.batch_size
num_samples = len(validation_generator.filenames)
validation_steps = num_samples // batch_size

model = unet(input_shape=(224, 224, 3), num_classes=3)

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=[MeanIoU(num_classes=3)])

model.fit(train_generator,
          validation_data=validation_generator,
          steps_per_epoch= steps_per_epoch,
          validation_steps= validation_steps,
          epochs=20,  # Treine o modelo por mais épocas
          verbose=1)