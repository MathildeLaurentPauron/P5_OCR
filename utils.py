import tensorflow_hub as hub
import tensorflow
from sklearn.preprocessing import MultiLabelBinarizer
import mlflow.sklearn
import pickle

# Charger le modèle USE depuis TensorFlow Hub
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Fonction pour obtenir l'embedding d'un texte avec USE
def get_use_embedding(question):
    return use_model([question]).numpy().flatten()

# Charger le modèle
model_uri = "mlruns/0/<run_id>/artifacts/embedding_type_multi_label_model"
model = mlflow.sklearn.load_model(model_uri)

#charger le mlb
mlb = pickle.load(open('mlb.pkl', 'rb'))


def predict_tags(question):
    # Conversion du texte en embedding, prétraitement ou vectorisation
    # Ici, supposons que text_to_embedding soit une fonction de prétraitement
    embedding = get_use_embedding(question)
    keywords = model.predict([embedding])
    predicted_tags = mlb.inverse_transform(keywords)
    return predicted_tags
    