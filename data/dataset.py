import os
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

# Divida o dataset estratificadamente em conjuntos de treinamento, validação e teste
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

# Em seguida, divida o conjunto de treinamento em treinamento e validação estratificadamente
train_image_paths, validation_image_paths, train_labels, validation_labels = train_test_split(
    train_image_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

# Agora, você tem três conjuntos (treinamento, validação e teste) com distribuição estratificada
# Certifique-se de que as proporções das classes estão equilibradas em todos os conjuntos


print("Tamanho do conjunto de treinamento:", len(train_image_paths))
print("Tamanho do conjunto de validação:", len(validation_image_paths))
print("Tamanho do conjunto de teste:", len(test_image_paths))