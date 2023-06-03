"""
Code du tri jour / nuit des images en fonction de l'heure de prise et des éphémérides
Code final du projet CODEV 110 : Catégorisation des images de la caméra plein ciel de l’observatoire de la Pointe du Diable
Iris AUBE, Eve BODOT, Julie ROLLET, Matias TRAN BINH
Mai 2023
"""

"""
Première partie de l'algorithme : Tri Jour / Nuit
"""
#importations nécessaires aux fonctions de ce module
import os
from datetime import datetime
import math
import shutil
import cv2
import numpy as np

"""
https://openclassrooms.com/forum/sujet/soleil-2
This simple function takes the date, time and location of any point on
the earth and return True for day and False for night.
:param when: Date and time in datetime format
:param longitude: Longitude in decimal degrees, east is positive
:param latitude: Latitude in decimal degrees, north is positive
:param twilight: optional twilight setting. Default='civil', None, 'nautical' or 'astronomical'.
:raises ValueError if twilight not recognised.
:returns boolean True = daytime (including twilight), False = nighttime.
This function is drawn from Jean Meeus' Astronomial Algorithms as
implemented by Michel J. Anders. In accordance with his Collective
Commons license, the reworked function is being released under the OSL
3.0 license by FDS as a part of the POLARIS project.
For FDM purposes, the actual time of sunrise and sunset is of no
interest, so function 12.6 is adapted to give just the day/night
decision, with allowance for different, generally recognised, twilight
tolerances.
FAA Regulation FAR 1.1 defines night as: "Night means the time between
the end of evening civil twilight and the beginning of morning civil
twilight, as published in the American Air Almanac, converted to local
time.
EASA EU OPS 1 Annex 1 item (76) states: 'night' means the period between
the end of evening civil twilight and the beginning of morning civil
twilight or such other period between sunset and sunrise as may be
prescribed by the appropriate authority, as defined by the Member State;
CAA regulations confusingly define night as 30 minutes either side of
sunset and sunrise, then include a civil twilight table in the AIP.
With these references, it was decided to make civil twilight the default.
Sources:
- http://www.esrl.noaa.gov/gmd/grad/solcalc/main.js
- http://www.esrl.noaa.gov/gmd/grad/solcalc/calcdetails.html
- http://michelanders.blogspot.co.uk/2010/12/calulating-sunrise-and-sunset-in-python.html
"""

def NomFichiers(files, directory):
    # Parcourir la liste des fichiers et renvoyer les noms de fichier

    # création d'une liste contenant le nom des dossiers
    liste_images = []
    # Une deuxième liste est créée avec les images dans le bon format pour être traitées par l'algorithme de ML (Machine Learning)
    liste_images_bon_format = []
    for file in files:
        liste_images.append(file)
        img = cv2.imread(os.path.join(directory, file))
        img = cv2.resize(img, (150, 150))  # Redimensionner les images si nécessaire
        liste_images_bon_format.append(img)
    return liste_images, liste_images_bon_format


def splitDate(date_string):
    # Transformation des noms pour un nouveau format datetime

    parts3 = date_string.split(".")
    parts = parts3[0].split("__")
    date = parts[0]
    heure = parts[1]

    parts1 = date.split("_")
    year = int(parts1[0])
    month = int(parts1[1])
    day = int(parts1[2])

    parts2 = heure.split("_")
    hour = int(parts2[0])
    minutes = int(parts2[1])
    seconds = int(parts2[2])

    return datetime(year, month, day, hour, minutes, seconds)


def listWhen(liste_images):
    # renvoie la liste des temps des photos dans le bon format
    LWhen = []
    for file in liste_images:
        LWhen.append(splitDate(file))
    return LWhen

def is_daytime_manuel (when):
    # Créée lorsque des faiblesses dans la fonction is_daytime ont été constatées en fin de projet
    # Obtention du mois de la date
    mois = when.month

    # Obtention de l'heure du lever et du coucher du soleil en fonction du mois et de la saison
    if mois==1:
        heure_lever_soleil = 7
        heure_coucher_soleil = 20
    elif mois==2:
        heure_lever_soleil = 7
        heure_coucher_soleil = 20
    elif mois==3: #Attention changement d'heure
        heure_lever_soleil = 6
        heure_coucher_soleil = 21
    elif mois==4:
        heure_lever_soleil = 6
        heure_coucher_soleil = 23
    elif mois==5:
        heure_lever_soleil = 5
        heure_coucher_soleil = 24
    elif mois==6:
        heure_lever_soleil = 4
        heure_coucher_soleil = 24
    elif mois==7:
        heure_lever_soleil = 5
        heure_coucher_soleil = 24
    elif mois==8:
        heure_lever_soleil = 5
        heure_coucher_soleil = 23
    elif mois==9:
        heure_lever_soleil = 6
        heure_coucher_soleil = 22
    elif mois==10: #Attention changement d'heure
        heure_lever_soleil = 6
        heure_coucher_soleil = 21
    elif mois==11:
        heure_lever_soleil = 7
        heure_coucher_soleil = 19
    elif mois==12:
        heure_lever_soleil = 7
        heure_coucher_soleil = 19

# Obtention de l'heure actuelle de la date
    heure_actuelle = when.hour

    # Vérification s'il fait jour ou nuit
    if heure_actuelle >= heure_lever_soleil and heure_actuelle < heure_coucher_soleil:
        return True  # Il fait jour
    else:
        return False  # Il fait nuit
    

def is_daytime(latitude, longitude, when, twilight):
    # fonction qui détermine si l'heure "when" est considéré de jour ou non
    # n'est pas utilisée dans la version rendue du code, mais pourra être exploitée si corrigée (moins d'incohérences entre octobre et mai)

    day = when.toordinal() - 693594
    t = when.time()
    time = (t.hour + t.minute / 60.0 + t.second / 3600.0) / 24.0

    # calculate julian day and century:
    jd = day + 2415018.5 + time
    jc = (jd - 2451545.0) / 36525.0

    # siderial time at greenwich
    gstime = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + (0.0003879331 - jc / 38710000) * jc ** 2) % 360.0

    # geometric mean longitude sun (deg)
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360

    # geometric mean anomaly sun (radians)
    m = math.radians(357.52911 + jc * (35999.05029 - 0.0001537 * jc))

    # sun equation of center
    sin1m, sin2m, sin3m = (math.sin(i * m) for i in range(1, 4))
    c = sin1m * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + sin2m * (0.019993 - 0.000101 * jc) + sin3m * 0.000289

    # calculate elements used in multiple places below:
    omega = math.radians(125.04 - 1934.136 * jc)
    latitude = math.radians(latitude)

    # mean obliquity of ecliptic, corrected (radians)
    seconds = 21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))
    e0 = 23.0 + (26.0 + seconds / 60.0) / 60.0
    e = math.radians(e0 + 0.00256 * math.cos(omega))

    # sun true longitude (deg)
    o = l0 + c

    # sun apparent longitude (radians)
    lambda_ = math.radians(o - 0.00569 - 0.00478 * math.sin(omega))

    # sun declination (radians)
    declination = math.asin(math.sin(e) * math.sin(lambda_))

    # sun right ascension (deg)
    rightasc = math.degrees(math.atan2(math.cos(e) * math.sin(lambda_), math.cos(lambda_)))

    elevation = math.degrees(math.asin(
        math.sin(latitude) * math.sin(declination) +
        math.cos(latitude) * math.cos(declination) *
        math.cos(math.radians(gstime + longitude - rightasc))))

    # - Solar diameter gives 0.833 degrees - rim of sun appears before centre of disc.
    # - For civil twilight, allow 6 degrees.
    # - For nautical twilight, allow 12 degrees.
    # - For astronomical twilight, allow 18 degrees.
    if twilight is None:
        limit = -0.8333
    elif twilight == 'civil':
        limit = -6.0
    elif twilight == 'nautical':
        limit = -12.0
    elif twilight == 'astronomical':
        limit = -18.0  
    else:
        raise ValueError('is_day() twilight argument must be one of: civil, nautical, astronomical or None.')

    return bool(elevation > limit) # true = day, false = night



def tri(LWhen, liste_images, liste_images_bon_format, latitude, longitude, twilight):
    # Tri des images en deux listes distinctes, récupère le nom de l'image
    images_jour = []
    images_nuit = []
    images_nuit_bon_format = []
    i = 0
    for i in range(0, len(LWhen)):
        if is_daytime_manuel(LWhen[i]):
            images_jour.append(liste_images[i])
        else:
            images_nuit.append(liste_images[i])
            images_nuit_bon_format.append(liste_images_bon_format[i])

    return images_jour, images_nuit, images_nuit_bon_format


def rangement(jour,nuit,directory, dossier_destination_Jour, dossier_destination_Nuit):
    ## Range les fichiers dans de nouveaux dossiers sur l'ordinateur (un pour les images de jour et l'autre pour les images de nuit)
    # dossier_destination_Jour et dossier_destination_Nuit : chemins vers les dossiers dans lesquels seront rangées les images à la suite du premier tri
    #Cette fonction n'est pas utilisée dans le code final pour ne pas créer de dossiers supplémentaires et charger la mémoire
   
    if not os.path.exists(dossier_destination_Jour):
        os.makedirs(dossier_destination_Jour)
    if not os.path.exists(dossier_destination_Nuit):
        os.makedirs(dossier_destination_Nuit)

    # Boucle à travers chaque fichier et les déplacer vers le dossier de destination
    for fichier_jour in jour:
        chemin_fichier_jour = directory + "/" + fichier_jour
        print(chemin_fichier_jour)
        shutil.move(chemin_fichier_jour, dossier_destination_Jour)

    for fichier_nuit in nuit:
        chemin_fichier_nuit = directory + "/" + fichier_nuit
        print(chemin_fichier_nuit)
        shutil.move(chemin_fichier_nuit, dossier_destination_Nuit)


def tri_jour_nuit(directory, latitude, longitude, twilight):
    #Fonction qui utilise les fonctions précédentes et sera appelée par le main
    # directory = Spécifiez le chemin d'accès au répertoire contenant les fichiers (attention à bien mettre / et pas \)

    # Utilisez la méthode listdir() de la bibliothèque os pour récupérer les noms des fichiers dans le répertoire
    files = os.listdir(directory)
    liste_images, liste_images_bon_format = NomFichiers(files, directory)
    #print("liste images au bon format : ", liste_images_bon_format)
    LWhen = listWhen(liste_images)
    images_jour, images_nuit, images_nuit_bon_format = tri(LWhen, liste_images, liste_images_bon_format, latitude, longitude, twilight)
    #print("images de nuit au bon format : ", images_nuit_bon_format)
    #print("images nuit : ", images_nuit)
    #print("images jour :" , images_jour)

    # converson de la liste des images de nuit au bon format en tableaux (pour le machine learning)
    images_nuit_bon_format = np.array(images_nuit_bon_format)

    print("Le tri jour / nuit est terminé")

    return liste_images,  images_jour, images_nuit, images_nuit_bon_format

"""
# bloc de test de la fonction is_daytime 
latitude = 48.3600473
longitude = -4.5705254
twilight = 'astronomical'
directory = 'C:/Users/neome/OneDrive/IMT Atlantique one drive/CODEV/ML Codev/Version finale/dossiers_tests/2023-05-19' #dossier contennat les images à trier
liste_images, images_jour, images_nuit, images_nuit_bon_format = tri_jour_nuit(directory, latitude, longitude, twilight)

print(is_daytime(longitude, latitude, splitDate('2023_05_20__03_05_57.jpg'), twilight))
"""