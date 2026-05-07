import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Configuration des dossiers
PATH = 'models/url'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def train_url_simple():
    print("🧠 Entraînement de l'IA URL (Version Finale)...")
    
    if not os.path.exists("dataset_url.csv"):
        print("❌ Erreur : dataset_url.csv introuvable.")
        return

    # 2. Chargement et Nettoyage forcé
    df = pd.read_csv("dataset_url.csv")
    
    # On nettoie les noms de colonnes et on réassigne le DataFrame pour être sûr
    df.columns = df.columns.str.strip().str.lower()
    
    # On crée des copies propres des colonnes pour éviter les problèmes d'index
    try:
        url_data = df['url'].astype(str).values
        label_data = df['label'].values
        print(f"✅ Données extraites avec succès ({len(url_data)} lignes).")
    except KeyError as e:
        print(f"❌ Erreur critique : La colonne {e} est introuvable malgré le nettoyage.")
        print(f"Colonnes disponibles : {list(df.columns)}")
        return

    # 3. Vectorisation par caractères
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
    X = vectorizer.fit_transform(url_data)
    y = label_data

    # 4. Split & Entraînement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Sauvegarde
    joblib.dump(model, os.path.join(PATH, 'phishing_url_model.pkl'))
    joblib.dump(vectorizer, os.path.join(PATH, 'url_vectorizer.pkl'))
    
    print(f"🚀 Succès ! Modèle URL sauvegardé.")
    print(f"📊 Précision : {model.score(X_test, y_test):.2%}")

if __name__ == "__main__":
    train_url_simple()