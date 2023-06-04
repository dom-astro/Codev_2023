# Codev_Git
Projet Codev Julie, Eve, Iris, Matias : Catégorisation d’images plein ciel de l’observatoire astronomique

# Liste des modules à importer :
import os
import cv2
import numpy as np
import math
# module des k plus proches voisins
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors

# modules pour le tri jour nuit
from datetime import datetime
from pandas import DataFrame

# modules non utilisés dans la dernière version du code, mais présents dans certaines fonctions 
import shutil (utile pour le déplacement de fichiers dans des dossiers)
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

