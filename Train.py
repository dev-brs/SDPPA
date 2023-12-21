import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Carregue o modelo pré-treinado MobileNetV2 (sem incluir as camadas densas superiores)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# Adicione camadas personalizadas para a tarefa de classificação
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)

# Camada de saída para a classificação
num_classes = 2  # Duas classes: "0" e "1"
classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_output')(x)

# Crie o modelo final
model = tf.keras.models.Model(inputs=base_model.input, outputs=classification_output)

# Congele as camadas do modelo base (opcional, se você quiser manter os pesos pré-treinados)
for layer in base_model.layers:
    layer.trainable = False

# Compile o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Carregue seus dados e pré-processe-os  NÃO FAZ NADA
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Carregue as imagens e converta-as em arrays numpy
images = []
labels = []  # Lista para armazenar as classes

# Diretórios de treinamento para as classes "0" e "1"
class_directories = ["0", "1"]

max_images_per_class = 1000  # Limite de 1000 imagens por classe

for class_dir in class_directories:
    class_path = os.path.join("data", class_dir)  # Substitua 'data' pelo caminho correto
    image_count = 0

    for image_path in os.listdir(class_path):
        if image_count >= max_images_per_class:
            break

        image = tf.keras.preprocessing.image.load_img(os.path.join(class_path, image_path), target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Pré-processamento
        images.append(image)
        labels.append(int(class_dir))  # Usar 0 para a classe "0" e 1 para a classe "1"

        image_count += 1

images = np.array(images)
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Divida os dados em conjunto de treinamento e validação (80% treinamento, 20% validação)
train_split = 0.8
split_idx = int(len(images) * train_split)

train_images = images[:split_idx]
train_labels = labels[:split_idx]

val_images = images[split_idx:]
val_labels = labels[split_idx:]

# Train the model with validation
history = model.fit(
    train_images,
    train_labels,
    epochs=20,
    batch_size=16,
    validation_data=(val_images, val_labels))

# Save the trained model
model.save("modelo_atualizado_v2.h5")

# Plot the training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.legend(['Treino', 'Validação'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')

# Save the graph as an image
plt.savefig('Gráfico_do_treino.png')

# Show the graph (optional)
plt.show()
