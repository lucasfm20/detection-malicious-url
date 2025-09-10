import pandas as pd



file_id = "1NezeELUkV621IkStglimCZORjlEKTKxH"
url = f"https://drive.google.com/uc?id={file_id}"

data = pd.read_csv(url)
data.tail()

data.isnull().sum()

data.type.value_counts()

rem = {"Category": {"benign": 0, "defacement": 1, "phishing":2, "malware":3}}
data['Category'] = data['type']
data = data.replace(rem)


# 0 não possuí e 1 possuí
data["possui_simbolos"] = data["url"].str.contains(r"[^a-zA-Z0-9:/.\-]", regex=True).astype(int)

feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
for a in feature:
    data[a] = data['url'].apply(lambda i: i.count(a))


data["https"] = data["url"].str.startswith("https://").astype(int)

from urllib.parse import urlparse


def verificaHost(url):
  host = urlparse(url).hostname
  if (host):
    return 1
  return 0

def tamanho_url(url):
  return len(url)



data['possui_host'] = data['url'].apply(lambda i: verificaHost(i))

data['tamanho_url'] = data["url"].apply(lambda i:tamanho_url(i))

data.head()


#Tratar outliers
antes = len(data)
Q1 = data['tamanho_url'].quantile(0.25)
Q3 = data['tamanho_url'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

data = data[(data['tamanho_url'] >= limite_inferior) & (data['tamanho_url'] <= limite_superior)]

depois = len(data)

# verificar se houve alteração
if antes != depois:
    print(f"Foram removidos {antes - depois} registros.")
else:
    print("Nenhum dado foi alterado/removido.")

    data = data.dropna(subset=["Category"])

X = data.drop(["url","type","Category"], axis=1)
y = data["Category"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

# Loop para treinar, prever e avaliar cada modelo
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(f"\n--- Modelo: {nome} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if nome == "RandomForest":
        joblib.dump(modelo, "RandomForest_phishing.pkl")
        print("RandomForest salvo como: RandomForest_phishing.pkl")