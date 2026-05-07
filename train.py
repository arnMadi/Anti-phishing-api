import pandas as pd
import re
import joblib # Plus robuste que pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- CONFIGURATION DES CHEMINS ---

PATH = 'models/url'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def clean_text(text):
    text = str(text).lower()
    # On garde les liens au lieu de les supprimer ? 
    # Pour du phishing, la structure des liens est un indice, 
    # mais pour un modèle texte pur, on les nettoie :
    text = re.sub(r"http\S+", "url", text) 
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_phishing_model():
    try:
        # 1. Chargement (vérifie bien que ton fichier s'appelle ainsi)
        if not os.path.exists("phishing_emails.csv"):
            print("❌ Erreur: Le fichier 'phishing_emails.csv' est introuvable !")
            return

        df = pd.read_csv("phishing_emails.csv")
        
        # Vérification des colonnes (ajuste selon ton dataset)
        # Si ton CSV a des noms différents, on les renomme :
        # df = df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})

        print(f"📦 Dataset chargé : {df.shape[0]} lignes.")

        # 2. Nettoyage
        print("🧹 Nettoyage des données...")
        df["text"] = df["text"].apply(clean_text)

        # 3. Vectorisation
        # On limite à 3000 features pour rester sous les 512Mo de Render
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]

        # 4. Entraînement
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("🧠 Entraînement du modèle (Logistic Regression)...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # 5. Sauvegarde au bon endroit pour app.py
        joblib.dump(model, "models/url/phishing_model.pkl")
        joblib.dump(vectorizer, "models/url/vectorizer.pkl")

        print("\n✅ SUCCÈS : Modèles sauvegardés dans le dossier /models/")
        print(f"Score de précision : {model.score(X_test, y_test):.2%}")

    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")

if __name__ == "__main__":
    train_phishing_model()