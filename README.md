# Medical AI Chatbot (RAG System)

Ce projet est un chatbot médical intelligent de bout en bout (End-to-End). Il utilise l'architecture **RAG (Retrieval-Augmented Generation)** pour fournir des réponses précises et contextuelles basées sur des documents médicaux de confiance, évitant ainsi les hallucinations des modèles d'IA standards.

## Fonctionnalités Clés

* **Récupération de Connaissance (Retrieval)** : Analyse des documents PDF médicaux pour extraire des informations pertinentes.
* **Base de Données Vectorielle** : Utilisation de **Pinecone** pour un stockage et une recherche sémantique ultra-rapide des connaissances.
* **Intelligence Artificielle** : Intégration de **Google Gemini 1.5 Flash** pour une génération de réponses naturelle et précise.
* **Interface Web Intuitive** : Une interface de chat fluide développée avec Flask pour une interaction utilisateur simplifiée.
* **Prompt Engineering** : Système de prompt optimisé pour garantir un comportement professionnel et sécurisé du chatbot.

## Architecture du Projet

Le système fonctionne selon un pipeline structuré :

1. **Chargement & Split** : Les documents médicaux sont découpés en "chunks" gérables.
2. **Embeddings** : Transformation du texte en vecteurs mathématiques.
3. **Stockage** : Indexation dans la base de données Pinecone.
4. **Inférence** : Lorsqu'une question est posée, le système récupère les 3 segments les plus proches sémantiquement avant de générer la réponse finale.

## 🛠️ Stack Technique

* **Backend** : Python, Flask.
* **LLM** : LangChain, Google Generative AI (Gemini).
* **Embeddings** : HuggingFace / LangChain Embeddings.
* **Vector Database** : Pinecone.
* **Parsing** : PyPDFLoader, RecursiveCharacterTextSplitter.

## Installation et Configuration

### 1. Cloner le projet

```bash
git clone https://github.com/votre-username/medical-chatbot-rag.git
cd medical-chatbot-rag

```

### 2. Configurer les variables d'environnement

Créez un fichier `.env` à la racine :

```env
PINECONE_API_KEY=votre_cle_pinecone
GOOGLE_API_KEY=votre_cle_gemini

```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt

```

### 4. Lancer l'application

```bash
python app.py

```

Accédez à l'interface sur `http://127.0.0.1:5000`.

## Structure du Code

* `app.py` : Point d'entrée de l'application Flask et orchestration des chaînes RAG.
* `src/helper.py` : Fonctions utilitaires pour le chargement des embeddings.
* `src/prompt.py` : Définition du `system_prompt` pour guider le comportement de l'IA.
* `templates/` : Interface utilisateur (HTML/JS).

---

## Rapport Technique

Un rapport technique approfondi est disponible pour ce projet. Il détaille :

* Le choix de la stratégie de découpage du texte (Chunking).
* L'évaluation de la pertinence des résultats de recherche Pinecone.
* La gestion de la latence entre le serveur Flask et l'API Gemini.

**[Télécharger le Rapport Technique (PDF)](./rapport_technique.pdf)**
## Licence

Ce projet est distribué sous la licence MIT.
