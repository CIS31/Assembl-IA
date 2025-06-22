# Assembl-IA

Présentation du projet
  
Présentation de la pipeline Azure  

![Demo](./assets/Fil%20rouge%20v2.png)

## Analyse textuelle

## Analyse textuelle

#### Présentation

Cette brique du projet vise à **détecter les émotions dans des textes en français** (transcriptions audio, commentaires, scripts, etc.).  
Afin d’aligner la sortie textuelle sur l’axe visuel, nous couvrons **7 émotions** : Tristesse (`sad`), Peur (`fear`), Colère (`anger`), Neutre (`neutral`), Surprise (`surprise`), Joie (`joy`) et Dégoût (`disgusted`).

#### Fonctionnalités

- ✅ Pré-traitement complet du texte (nettoyage, normalisation, tokenisation)
- ✅ Classification des émotions sur 7 classes
- ✅ Export CSV contenant : timestamp, texte original, émotion prédite, score de confiance
- ✅ Intégration directe dans la pipeline Azure (Databricks + Blob Storage + Postgres)

#### Pipeline Azure

1. Lecture des fichiers texte ou transcriptions stockés dans le **Blob Storage**  
2. **Databricks** appelle le notebook de prédiction NLP  
3. Les prédictions sont :
   - stockées au format CSV dans le Blob Storage (dossier *output*)  
   - insérées dans **Postgres** pour exploitation BI  
4. Les métriques d’exécution (latence, nombre de tokens) sont remontées à **Azure Application Insights**

#### Modèle utilisé

| Nom | Base | Type | Lien |
|-----|------|------|------|
| `assembl-ia/french_emotion_camembert-7cls` | CamemBERT-base | Fine-tune (7 émotions) | 🔗 [Hugging Face (original)](https://huggingface.co/astrosbd/french_emotion_camembert) |

- **Fine-tuning** réalisé sur un jeu de données équilibré de **5 033 phrases** (719 exemples/émotion).  
- **Epochs** : 5   •  **Batch size** : 16   •  **LR** : 2e-5

#### Données d’entraînement

| Source | Langue | Taille | Particularité |
|--------|--------|--------|---------------|
| **EMODIFT** | FR | 3 194 | Annoté manuellement |
| **TPM-28 / emotion-FR** | FR | 1 839 | Contient la classe *Dégoût* |

Les deux jeux ont été fusionnés puis ré-équilibrés pour obtenir exactement **719 exemples / classe**.

#### Évaluation du modèle

| Emotion   | Précision | Rappel | F1-score |
|-----------|-----------|--------|----------|
| Sad       | 0.94      | 0.95   | 0.94 |
| Fear      | 0.91      | 0.87   | 0.89 |
| Anger     | 0.87      | 0.88   | 0.88 |
| Neutral   | 0.86      | 0.86   | 0.86 |
| Surprise  | 0.99      | 0.98   | 0.99 |
| Joy       | 0.93      | 0.86   | 0.90 |
| Disgusted | 0.83      | 0.93   | 0.88 |
| **Macro avg** | **0.91** | **0.90** | **0.90** |
| **Accuracy globale** |        |        | **0.90** |

![Confusion Matrix](./assets/confusion_matrix_text_model.png)

> Le score de 90 % sur le set de test confirme l’intérêt du fine-tuning pour ajouter la classe *Dégoût* (vs 82 % avant adaptation).


## Analyse audio

## Analyse vidéo

#### Présentation

Il s'agit d'un gif, la vidéo au format .mp4 est disponible dans le dossier output

![Demo](./assets/video_vitrine.gif)


##  Fonctionnalités

- ✅ Lecture vidéo frame par frame
- ✅ Détection des visages
- ✅ Si visage assez grand → Détection des émotions (les 2 classes majoritaires)
- ✅ Annotation des résultats sur la vidéo en output
- ✅ Création d'un timeline (fichier CSV)

## Pipeline Azure

- ✅ Lecture des variables d'environnement contenues dans les paramètres du job databricks
- ✅ Recupération de la dernière vidéo présente sur le blob storage
- ✅ Traitement
- ✅ Enregistrement de la vidéo annotée et de la timeline dans le blob storage
- ✅ Enregistrement de la timeline dans postgres

##  Modèles utilisés

- YOLO v8 : 
🔗 https://yolov8.com/

- facial_emotions_image_detection : 
🔗 https://huggingface.co/dima806/facial_emotions_image_detection

##  Evaluation des modèles 

- utilisation du dataset de test suivant :
🔗 https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

- resultats : 
![Demo](./assets/testdumodelvideo.png)
  
| Emotion   | Précision |
|-----------|-----------|
| Angry     | 0.772     |
| Disgust   | 1.000     |
| Fear      | 0.838     |
| Happy     | 0.822     |
| Neutral   | 0.740     |
| Sad       | 0.943     |
| Surprise  | 0.928     |

Précision globale : **0.847**

