"""
Code de classification des images de nuit avec la méthodes des k plus proches voisins (machine learning)
Code final du projet CODEV 110 : Catégorisation des images de la caméra plein ciel de l’observatoire de la Pointe du Diable
Iris AUBE, Eve BODOT, Julie ROLLET, Matias TRAN BINH
Mai 2023
"""

"""
Deuxième partie de l'algorithme : Catégorisation des images avec méthodes de Machine Learning
Méthode des K plus proches voisins (version sans prétraitements)
"""

#importations nécessaires au projet
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps



def apply_circular_mask(image):
    # Fonction d'application du masque

    # Récupérer la taille de l'image
    width, height = 150, 150

    # Calculer le rayon du cercle à dessiner
    radius = min(width, height) // 2

    # Créer une nouvelle image blanche avec un canal alpha
    img_mask = Image.new('RGBA', (width, height), (255, 255, 255, 0))

    # Dessiner un cercle blanc à l'intérieur de l'image blanche
    draw = ImageDraw.Draw(img_mask)
    draw.ellipse((width/2 - radius, height/2 - radius, width/2 + radius, height/2 + radius), fill=(255, 255, 255, 255))

    # Appliquer le masque à l'image originale
    img_mask.paste(image, (0, 0), mask=img_mask)

    return img_mask



def chargement_images_bdd(degage_dir, couverture_partielle_dir,couverture_totale_dir) :
    # Chargement des images et création des étiquettes correspondantes
    images = []
    labels = []

    # Chargement des images de 'degage'
    for filename in os.listdir(degage_dir):
        img = cv2.imread(os.path.join(degage_dir, filename))
        img = cv2.resize(img, (150, 150))  # Redimensionner les images si nécessaire
        images.append(img)
        labels.append('degage')

    # Chargement des images de 'couverture_partielle'
    for filename in os.listdir(couverture_partielle_dir):
        img = cv2.imread(os.path.join(couverture_partielle_dir, filename))
        img = cv2.resize(img, (150, 150))  # Redimensionner les images si nécessaire
        images.append(img)
        labels.append('couverture_partielle')

    # Chargement des images de 'couverture_totale'
    for filename in os.listdir(couverture_totale_dir):
        img = cv2.imread(os.path.join(couverture_totale_dir, filename))
        img = cv2.resize(img, (150, 150))  # Redimensionner les images si nécessaire
        images.append(img)
        labels.append('couverture_totale')

    # Conversion des listes en tableaux numpy
    images = np.array(images)
    labels = np.array(labels)

    print("Le chargement des images de la base de données est terminé")

    return images, labels



# code utilisé pour la phase de test pendant le développement du projet qui pourrait être utile 
"""
# ANNEXE
#bloc de code pour tracer l'erreur en fonction de k, choisir le k tel que l'erreur est minimale
errors = []
for k in range(2,30):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(X_train, y_train).score(X_test, y_test)))
plt.plot(range(2,15), errors, 'o-')
plt.show()

# Séparation des données en ensembles d'entraînement et de test
# test_size (compris entre 0 et 1) est la proportion de la base de donnéees réservée aux tests, 
# ces données ne seront pas utilisées pour l'entraînement, car un modèle de machine learning est toujours meilleur sur des données déjà vues
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Calcul de l'exactitude (accuracy) du modèle 
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude : {:.2f}%".format(accuracy * 100))
"""