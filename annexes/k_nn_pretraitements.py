"""
Version du code avec plus de prétraitements (masque, transformée de Fourrier, noir et blanc)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Définition des chemins vers les dossiers contenant les images
degage2_dir = '/chemin/vers/degage2'
couverture_partielle2_dir = '/chemin/vers/couverture_partielle2'
couverture_totale2_dir = '/chemin/vers/couverture_totale2'

# Chargement des images et création des étiquettes correspondantes
images = []
labels = []

def apply_circular_mask(image):
    # Récupérer la taille de l'image
    width, height = image.size

    # Calculer le rayon du cercle à dessiner
    radius = min(width, height) // 1.8

    # Créer une nouvelle image blanche avec un canal alpha
    img_mask = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Dessiner un cercle blanc à l'intérieur de l'image blanche
    draw = ImageDraw.Draw(img_mask)
    draw.ellipse((width/2 - radius, height/2 - radius, width/2 + radius, height/2 + radius), fill=(255, 255, 255, 255))

    # Appliquer le masque à l'image originale
    img_mask.paste(image, (0, 0), mask=img_mask)

    return img_mask

# Chargement des images de 'degage2'
for filename in os.listdir(degage2_dir):
    img = Image.open(os.path.join(degage2_dir, filename))
    img = img.convert('RGBA')
    img = apply_circular_mask(img)  # Appliquer le cache circulaire
    img = img.convert('L')  # Conversion en niveau de gris
    img = np.array(img)
    #img_lbp = extract_lbp_features(img)  # Extraction de caractéristiques LBP
    #images.append(img_lbp)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum /= img.shape[0] * img.shape[1]
    images.append(magnitude_spectrum)
    labels.append('degage')

# Chargement des images de 'couverture_partielle2'
for filename in os.listdir(couverture_partielle2_dir):
    img = Image.open(os.path.join(couverture_partielle2_dir, filename))
    img = img.convert('RGBA')
    img = apply_circular_mask(img)  # Appliquer le cache circulaire
    img = img.convert('L')  # Conversion en niveau de gris
    img = np.array(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum /= img.shape[0] * img.shape[1]
    images.append(magnitude_spectrum)
    labels.append('couverture_partielle')

# Chargement des images de 'couverture_totale2'
for filename in os.listdir(couverture_totale2_dir):
    img = Image.open(os.path.join(couverture_totale2_dir, filename))
    img = img.convert('RGBA')
    img = apply_circular_mask(img)  # Appliquer le cache circulaire
    img = img.convert('L')  # Conversion en niveau de gris
    img = np.array(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum /= img.shape[0] * img.shape[1]
    images.append(magnitude_spectrum)
    labels.append('couverture_totale')

# Conversion des listes en tableaux numpy
images = np.array(images)
labels = np.array(labels)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Aplatir les images en vecteurs
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Création du classifieur k-NN et ajustement sur les données d'entraînement
k = 5  # Nombre de voisins
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = knn.predict(X_test)

# Calcul de l'exactitude (accuracy) du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude : {:.2f}%".format(accuracy * 100))


