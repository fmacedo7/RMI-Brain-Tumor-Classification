import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

dataset_root = 'data/dataset'

image_paths = []
labels = []

for class_folder in os.listdir(dataset_root):
    class_folder_path = os.path.join(dataset_root, class_folder)
    if os.path.isdir(class_folder_path):
        # Loop através das imagens dentro de cada pasta de classe
        for image_file in os.listdir(class_folder_path):
            if image_file.endswith('.png'):  # Verifico a extensão correta das imagens (PNG)
                image_path = os.path.join(class_folder_path, image_file)
                image_paths.append(image_path)
                labels.append(int(class_folder))  # Usei o número da pasta como rótulo

# Divide o dataset em conjuntos de treinamento e teste
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)

# # Posso ver o caminho e o rotulo das imagens armazenadas na variavel de treinamento
# print("Primeiros caminhos das imagens de treinamento:", train_image_paths[:5])
# print("Rótulos de treinamento correspondentes:", train_labels[:5])

# # Posso ver a quantidade de imagens de treinamento e o rotulo correspondente
# print("Número de imagens de treinamento:", len(train_image_paths))
# print("Número de rótulos de treinamento:", len(train_labels))

# # Aqui me mostra qual e a imagen armazenada
# for img_path in train_image_paths[:5]:
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()