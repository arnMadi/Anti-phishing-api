import requests
from bs4 import BeautifulSoup
import re

def get_site_content(url):
    """
    Explore une URL et extrait le titre et le texte principal.
    """
    try:
        # 1. On définit un "User-Agent" pour simuler un vrai navigateur (évite les blocages)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 2. Tentative de récupération de la page
        # On ajoute un timeout de 5 secondes pour ne pas faire ramer l'API
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status() # Lève une erreur si le site est inaccessible (404, 500, etc.)

        # 3. Analyse du contenu HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Supprimer les balises inutiles pour l'analyse (scripts, styles)
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()

        # 4. Extraction du titre et du texte
        title = soup.title.string if soup.title else "Sans titre"
        main_text = soup.get_text(separator=' ', strip=True)

        # 5. Nettoyage simple (enlever les espaces en trop)
        clean_text = re.sub(r'\s+', ' ', main_text)

        # On combine le titre et le texte pour donner plus de contexte à l'IA
        final_content = f"{title} {clean_text}"
        
        return {
            "success": True,
            "content": final_content[:2000], # On limite à 2000 caractères pour Render (mémoire)
            "status_code": response.status_code
        }

    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "content": ""
        }

# --- ZONE DE TEST RAPIDE ---
if __name__ == "__main__":
    url_a_tester = "https://www.google.com"
    print(f"🔎 Test du scraper sur : {url_a_tester}")
    
    resultat = get_site_content(url_a_tester)
    
    if resultat["success"]:
        print("✅ Succès !")
        print(f"Contenu extrait (début) : {resultat['content'][:150]}...")
    else:
        print(f"❌ Échec : {resultat['error']}")