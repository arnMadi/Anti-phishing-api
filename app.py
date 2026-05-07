from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION DES CHEMINS (Basée sur ton image) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèles TEXTE
TEXT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'text', 'phishing_text_model.pkl')
TEXT_VECT_PATH = os.path.join(BASE_DIR, 'models', 'text', 'text_vectorizer.pkl')

# Modèles URL
URL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'url', 'phishing_url_model.pkl')
URL_VECT_PATH = os.path.join(BASE_DIR, 'models', 'url', 'url_vectorizer.pkl')

# --- CHARGEMENT DES MODÈLES ---
print("⌛ Chargement des modèles IA...")
try:
    text_model = joblib.load(TEXT_MODEL_PATH)
    text_vect = joblib.load(TEXT_VECT_PATH)
    url_model = joblib.load(URL_MODEL_PATH)
    url_vect = joblib.load(URL_VECT_PATH)
    print("✅ Tous les modèles sont chargés avec succès.")
except Exception as e:
    print(f"❌ ERREUR DE CHARGEMENT : {e}")
    text_model = text_vect = url_model = url_vect = None

@app.route('/analyze', methods=['POST'])
def analyze():
    # Sécurité : vérifier si les modèles sont bien là
    if not all([text_model, text_vect, url_model, url_vect]):
        return jsonify({"error": "Le serveur n'a pas pu charger les modèles .pkl"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Requête invalide (JSON manquant)"}), 400

    user_text = data.get('text')
    user_url = data.get('url')

    if not user_text and not user_url:
        return jsonify({"error": "Fournir au moins un champ 'text' ou 'url'"}), 400

    results = {}

    # --- ANALYSE TEXTE ---
    if user_text:
        try:
            vectorized = text_vect.transform([str(user_text)])
            proba = text_model.predict_proba(vectorized)[0][1]
            results['text'] = {
                "score": round(float(proba) * 100, 2),
                "is_phishing": bool(proba > 0.5),
                "verdict": "Suspect" if proba > 0.5 else "Sûr"
            }
        except Exception as e:
            results['text'] = {"error": str(e)}

    # --- ANALYSE URL ---
    if user_url:
        try:
            # On utilise le vectorizer par caractères (Option Simple)
            vectorized = url_vect.transform([str(user_url)])
            proba = url_model.predict_proba(vectorized)[0][1]
            results['url'] = {
                "score": round(float(proba) * 100, 2),
                "is_phishing": bool(proba > 0.5),
                "verdict": "Malveillant" if proba > 0.5 else "Sûr"
            }
        except Exception as e:
            results['url'] = {"error": str(e)}

    # Calcul de l'alerte globale
    is_danger = any(res.get('is_phishing', False) for res in results.values() if isinstance(res, dict))

    return jsonify({
        "status": "success",
        "analysis": results,
        "global_alert": is_danger,
        "recommendation": "⚠️ Prudence recommandée !" if is_danger else "✅ Aucun danger détecté."
    })

if __name__ == '__main__':
    # On tourne sur le port 5000 par défaut
    app.run(debug=True, port=5000)