"""
Code final du projet CODEV 110 : Catégorisation des images de la caméra plein ciel de l’observatoire de la Pointe du Diable
Iris AUBE, Eve BODOT, Julie ROLLET, Matias TRAN BINH
Mai 2023
"""

#importation des 2 fichiers contenant les fonctions utiles au projet
from tri_jour_nuit_final import *
from tri_knn_final import *

#importations nécessaires au projet
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import matplotlib.pyplot as plt
from datetime import datetime
import math
import shutil
from pandas import DataFrame

#import joblib   
# # permet de sauvegarder le modèle d'apprentissage automatique pour ne pas le rééxecuter à chaque fois 
# cf annexes


# chemins vers les dossiers, tri jour nuit
#dossier contenant les images à trier
directory = 'chemin/absolu/vers/dossier' 

"""
# les deux dossiers suivants ne sont pas utilisés dans la dernière version du code (fichier csv préféré par le client)
dossier_destination_Jour = "chemin/vers/dossier/jour" #dossier où seront rangées les images de jour
dossier_destination_Nuit = "chemin/vers/dossier/nuit" #dossier où seront rangées les images de nuit
"""

# Définition des chemins vers les dossiers contenant les images d'entraînement, tri machine learning
degage_dir = 'chemin/absolu/vers/dossier/degage'
couverture_partielle_dir = 'chemin/absolu/vers/dossier/couverture_partielle'
couverture_totale_dir = 'chemin/absolu/vers/dossier/couverture_totale'


# Utilisation des heures de capture d'image pour savoir s'il fait jour (True) ou nuit (False)
# en prennant en compte la nuit comme environ 1h30 après le coucher du Soleil
latitude = 48.3600473
longitude = -4.5705254
twilight = 'astronomical'

# Nombre de voisins, déterminé préalablement sur la base de données pour obtenir un résultat optimal
# (k éventuellement à modifier si la base de données d'entraînement change)
#un bloc de code permettant de déterminer k est fourni à la fin du fichier "tri_knn_final.py"
k = 20 



"""
Exécution des deux parties de l'algorithme
"""

print("Lancement du programme")
# premier tri : obtention de la liste avec les images de nuit
liste_images, images_jour, images_nuit, images_nuit_bon_format = tri_jour_nuit(directory, latitude, longitude, twilight)

# Liste des images à trier (les images de nuit)
X_test = images_nuit_bon_format

# chargement des images pour l'entraînement du modèle
images, labels = chargement_images_bdd(degage_dir, couverture_partielle_dir,couverture_totale_dir)
X_train = images
y_train = labels

# Aplatir les images en vecteurs
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Création du classifieur k-NN et ajustement sur les données d'entraînement
knn = KNeighborsClassifier(n_neighbors=k,)
knn.fit(X_train, y_train)
print("Le modèle est créé")

# Prédiction sur les données de test
y_pred = knn.predict(X_test)


def ecriturecsv2(liste_images, nuit, labels_categorie):
    """
    liste_images : liste de toutes les image à trier (dossier initial)
    nuit : liste ordonnée des images de nuit 
    labels_categorie : liste ordonnée des labels prédits par le modèle de machine learning pour les images de nuit
    """
    
    jour_nuit=[] # liste ordonnée des labels "jour" et "nuit" 
    categories_nuit=[] # liste ordonnée des labels de classificataion "degage", "partiellement_couvert" et "totalement_couvert"
    # remarque : dans categories_nuit des éléments nuls sont ajoutés aux indexes des images de jour

    j=0 # indice de parcout de la liste des labels_catégorie pour les images de nuit

    for i in liste_images: 
        if i in nuit:
            jour_nuit.append("nuit")
            categories_nuit.append(labels_categorie[j])
            j+=1
        else : 
            jour_nuit.append("jour")
            categories_nuit.append("")

    # création d'un fichier csv dans le repertoire courant
    Data={"images":liste_images, "jour/nuit":jour_nuit, "categorie de nuit": categories_nuit}
    donnees = DataFrame(Data, columns= ["images", "jour/nuit", "categorie de nuit"])
    donnees.to_csv ('resultat19_test.csv', index = None, header=True, encoding='utf-8', sep=';')
    print("Fichier csv créé !")
     

# création du fichier csv final :
ecriturecsv2(liste_images, images_nuit, y_pred)


