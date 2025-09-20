

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    """Entrena un modelo de clasificación de texto y guarda los archivos necesarios."""
    
    # Cargar los datos
    print("Cargando datos...")
    dataframe = pd.read_csv(
        "files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )
    
    print(f"Datos cargados: {len(dataframe)} ejemplos")
    print(f"Distribución de clases: {dataframe['target'].value_counts().to_dict()}")
    
    # Usar todos los datos para entrenar
    X = dataframe['phrase']
    y = dataframe['target']
    
    print(f"Entrenando con todos los datos: {len(X)} ejemplos")
    
    # Crear y entrenar el vectorizador TF-IDF
    print("Entrenando vectorizador TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),  # Unigramas y bigramas
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_tfidf = vectorizer.fit_transform(X)
    
    print(f"Dimensiones de la matriz TF-IDF: {X_tfidf.shape}")
    
    # Crear y entrenar el clasificador
    print("Entrenando clasificador...")
    clf = LogisticRegression(
        random_state=42,
        max_iter=3000,
        C=1.0,
        solver='lbfgs'
    )
    
    clf.fit(X_tfidf, y)
    
    # Evaluar el modelo en todo el dataset
    y_pred = clf.predict(X_tfidf)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Precisión en todo el dataset: {accuracy:.4f}")
    
    # Guardar el vectorizador
    print("Guardando vectorizador...")
    with open("homework/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
    
    # Guardar el clasificador
    print("Guardando clasificador...")
    with open("homework/clf.pickle", "wb") as file:
        pickle.dump(clf, file)
    
    print("¡Entrenamiento completado!")
    print("Archivos guardados:")
    print("- homework/vectorizer.pkl")
    print("- homework/clf.pickle")
    
    return clf, vectorizer, accuracy


if __name__ == "__main__":
    train_model()