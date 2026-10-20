import re
from agent import ManagerAgent

def test_agent():
    question = (
       'In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?'
    )
    
    print("--- Envoi de la question au Manager ---")
    try:
        # Lancement de l'agent
        agent = ManagerAgent()
        
        output = agent(question)
        print("\n--- Sortie brute de l'agent ---")
        print(output)
        
        # Test du parser de réponse
        match = re.search(r"(?i)FINAL\s*ANSWER\s*:\s*(.*)", str(output))
        if match:
            print(f"\n✅ Succès ! Réponse extraite : {match.group(1).strip()}")
        else:
            print("\n⚠️ Attention : L'agent a répondu mais la balise 'FINAL ANSWER:' est manquante ou mal formatée.")
            
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution : {e}")

if __name__ == "__main__":
    test_agent()