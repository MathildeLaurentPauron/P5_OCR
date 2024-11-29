import tensorflow_hub as hub
import tensorflow
from sklearn.preprocessing import MultiLabelBinarizer
import mlflow.sklearn
import pickle

# Téléchargement et chargement du binariseur depuis MLflow
artifact_dir = mlflow.artifacts.download_artifacts(artifact_uri="runs:/722883fac0e649de8e7c75fafcfdeb43/mlb.pkl")
local_binarizer_path = os.path.join(artifact_dir, "mlb.pkl")
with open(local_binarizer_path, "rb") as f:
    mlb = pickle.load(f)

# Charger le modèle USE depuis TensorFlow Hub
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Fonction pour obtenir l'embedding d'un texte avec USE
def get_use_embedding(question):
    return use_model([question]).numpy().flatten()

# Charger le modèle
model = mlflow.sklearn.load_model("runs:/722883fac0e649de8e7c75fafcfdeb43/<embedding_type>_multi_label_model")

def predict_tags(question):
    embedding = get_use_embedding(question)
    keywords = model.predict([embedding])
    predicted_tags = mlb.inverse_transform(keywords)
    return predicted_tags
    