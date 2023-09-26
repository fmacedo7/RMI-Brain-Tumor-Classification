from tensorflow.python.keras.metrics import MeanIoU
from tensorflow.python.keras.optimizer_v1 import Adam
from keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append('/Users/feliz/OneDrive/Documentos/UESPI/8_bloco/tcc_2/codigo')
from src.models import unet

# Crie geradores de dados para treinamento e validação
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalização das imagens
    # Outros parâmetros de aumento de dados, se aplicável
)

validation_data_generator = ImageDataGenerator(
    rescale=1.0 / 255  # Normalização das imagens de validação
    # Outros parâmetros, se neclecessário
)

train_generator = train_data_generator.flow_from_directory(
    'caminho_para_dados_de_treinamento',
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # Para imagens de entrada
    classes=['1', '2', '3']  # Adicione as classes correspondentes às pastas
)

validation_generator = validation_data_generator.flow_from_directory(
    'caminho_para_dados_de_validacao',
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # Para imagens de entrada
    classes=['1', '2', '3']  # Adicione as classes correspondentes às pastas
)


model = unet(input_shape=(224, 224, 3), num_classes=3)  # Certifique-se de ajustar o número de classes conforme seu projeto

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=[MeanIoU(num_classes=3)])

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=10,  # Ajuste o número de épocas conforme necessário
          verbose=1)