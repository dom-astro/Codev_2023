import joblib

# Supposons que vous avez votre modèle KNN entraîné dans une variable appelée "knn_model"
joblib.dump(knn_model, 'chemin/vers/votre/modele_knn.pkl')

# Charger le modèle à partir du fichier enregistré
knn_model = joblib.load('chemin/vers/votre/modele_knn.pkl')

# Utiliser le modèle pour prédire la catégorie des nouvelles images
nouvelles_images = ...  # Chargez vos nouvelles images ici
predictions = knn_model.predict(nouvelles_images)