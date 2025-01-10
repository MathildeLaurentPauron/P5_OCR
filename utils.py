# import tensorflow_hub as hub
# import tensorflow
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer 
import pickle
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

# Fonction de nettoyage de texte
def clean_text(text):
    text = re.sub(r'[^\w\s#+]', '', text)  # Enlever la ponctuation
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'<[^>]+>', '', text)  # Enlever les balises HTML
    tokens = nltk.word_tokenize(text)  # Tokenisation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Suppression des stopwords
    return ' '.join(tokens)

# Téléchargement et chargement du binariseur depuis MLflow
binarizer_path = "mlb.pkl"
with open(binarizer_path, "rb") as f:
    mlb = pickle.load(f)

# Feature extraction with Bag-of-Words
def get_bow_embedding(question):
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return vectorizer.transform([clean_text(question)]).toarray()

# Charger le modèle
model_path = "model_BoW.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict_tags(question):
    embedding = get_bow_embedding(question)
    keywords = model.predict(embedding)
    predicted_tags = mlb.inverse_transform(keywords)
    return predicted_tags
    