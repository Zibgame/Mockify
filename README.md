Voici un fichier `README.md` complet, clair et structuré pour documenter votre script de génération automatique de mockups Kindle.

```markdown
# Mockify — Kindle Case Mockup Generator

Mockify est un outil en ligne de commande (CLI) écrit en Python qui permet d'appliquer automatiquement une texture artistique ou un design sur différents modèles (templates) de coques Kindle en utilisant l'API OpenAI. 

Le script intègre une boucle d'assurance qualité (QC) via GPT-4o Vision pour valider le rendu visuel et corriger dynamiquement les erreurs de composition constatées lors d'un premier essai.

---

## 🚀 Fonctionnalités

* **Gestion Multi-Mockup** : Prend en charge nativement 4 types de présentations (Vue simple, double, étui ouvert, coque transparente/blanche).
* **Contrôle Qualité Automatique (QC)** : Utilise la puissance d'analyse visuelle de `gpt-4o` pour détecter les anomalies (proportions étirées, débordements, mauvaise application).
* **Correction Dynamique Adaptative** : En cas d'échec au contrôle qualité, le script génère automatiquement des instructions de correction ciblées et relance une génération améliorée.
* **Interface CLI Interactive** : Menu de sélection des mockups à générer, barre de progression visuelle, indicateurs de coûts réels/estimés et compte à rebours interactif en cas d'erreur.

---

## 📁 Structure des Dossiers Recommandée

Pour que le script fonctionne correctement, votre projet doit respecter l'arborescence suivante :

```text
📂 mon-projet/
├── 📄 .env                       # Clé API OpenAI
├── 📂 src/
│   └── 📄 main.py                # Le script principal (votre code)
└── 📂 assets/
    ├── 📂 mockups/               # Templates originaux à modifier
    │   ├── 🖼️ basic.png
    │   ├── 🖼️ double.png
    │   ├── 🖼️ open.png
    │   └── 🖼️ clear.png
    └── 📂 prompts/               # Configuration des instructions textuelles
        ├── 📄 texture_applyer.txt # Prompt système de base pour le rendu
        └── 📄 ImageChecker.txt    # Prompt système pour l'analyse de qualité

```

---

## ⚙️ Installation et Configuration

### 1. Prérequis

Assurez-vous d'avoir installé **Python 3.8+** ainsi que le SDK officiel OpenAI :

```bash
pip install openai

```

### 2. Configuration des Variables d'Environnement

Créez un fichier nommé `.env` à la racine de votre projet et ajoutez-y votre clé secrète OpenAI :

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

```

### 3. Fichiers de Prompts

Assurez-vous de remplir les fichiers de configuration situés dans `assets/prompts/` :

* `texture_applyer.txt` : Décrit comment l'IA doit manipuler l'image (ex: "Prends l'image 2 et applique-la de manière réaliste sur la coque de l'image 1...").
* `ImageChecker.txt` : Doit demander à `gpt-4o` de renvoyer un format JSON strict contenant obligatoirement les clés `has_problem`, `problems`, `severity`, et `details`.

---

## 💻 Utilisation

Pour lancer la génération, exécutez le script en lui passant le chemin de votre fichier de texture (le design à appliquer) en argument :

```bash
python src/main.py /chemin/vers/ma_texture.png

```

### Déroulement de l'exécution :

1. **Sélection des mockups** : Le script affiche la liste des mockups disponibles. Tapez les numéros correspondants séparés par un espace (ex: `1 3`) ou appuyez sur **Entrée** pour tous les générer.
2. **Génération & Validation** : Pour chaque mockup sélectionné, l'image est générée, enregistrée dans le dossier `outputs/`, puis scannée par le module QC.
3. **Gestion des Échecs (Interactif)** : Si un défaut est détecté, un compte à rebours de 5 secondes se lance :
* Pressez `S` pour ignorer et conserver l'image défectueuse.
* Pressez `R` (ou attendez la fin du décompte) pour **relancer automatiquement** une génération corrigée.


4. **Résumé** : Un tableau récapitulatif affiche le temps global, les statuts de validation et le coût financier réel calculé depuis les tokens consommés.

---

## 🛠️ Spécifications Techniques

* **Génération d'images** : Propulsé par le modèle `gpt-image-2` (via la méthode `client.images.edit`).
* **Analyse Vision** : Gérée par le modèle `gpt-4o` configuré pour lire des structures JSON à l'aide de blocs de code markdown nettoyés à la volée.
* **Format de sortie** : Images au format PNG de dimensions `1024x1024` nommées de façon unique : `{nom_texture}_{nom_mockup}_{timestamp}.png`.

```

```
