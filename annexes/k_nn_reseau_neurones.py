"""
Tests avec une méthode basée sur les réseaux de neuronnes. 
Utilisation d'un module différent que pour la méthode des k plus proches voisins.
Cette piste est intéressante, mais n'a pas été beaucoup développée dans les limites de ce projet.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Définition des chemins vers les dossiers contenant les images
train_dir = '/chemin/vers/degage'
val_dir = '/chemin/vers/couverture_partielle'
test_dir = '/chemin/vers/couverture_totale'

# Définition des paramètres du modèle
batch_size = 32
epochs = 10
img_height, img_width = 150, 150
num_classes = 3  # Nombre de classes (dégagé, partiellement couvert, totalement nuageux)

# Création des générateurs de données pour la formation, la validation et les tests
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Création du modèle CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(test_generator)
print("Loss:", loss)
print("Accuracy:", accuracy)