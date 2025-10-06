import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

file_id = "1NezeELUkV621IkStglimCZORjlEKTKxH"
url = f"https://drive.google.com/uc?id={file_id}"

data = pd.read_csv(url)

#Funções
def verificaHost(url):
  host = urlparse(url).hostname
  if (host):
    return 1
  return 0

def tamanho_url(url):
  return len(url)

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

#Tratar outliers
def trataOutlier(data):
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
    return data

counts = data.type.value_counts()
counts_df = counts.reset_index()
counts_df.columns = ['type', 'count']

plt.figure(figsize=(7,5))
sns.barplot(data=counts_df, x='type', y='count', palette='pastel')

plt.title('Quantidade de tipos')
plt.xlabel('Tipo')
plt.ylabel('Quantidade')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

rem = {"Categoria": {"benign": 0, "defacement": 1, "phishing":2, "malware":3}}
data['Categoria'] = data['type']
data = data.replace(rem)

data["possui_simbolos"] = data["url"].str.contains(r"[^a-zA-Z0-9:/.\-]", regex=True).astype(int)

feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
for a in feature:
    data[a] = data['url'].apply(lambda i: i.count(a))

data["https"] = data["url"].str.startswith("https://").astype(int)
data['possui_host'] = data['url'].apply(lambda i: verificaHost(i))
data['tamanho_url'] = data["url"].apply(lambda i:tamanho_url(i))
data['quanti_inteiros']= data['url'].apply(lambda i: digit_count(i))
data['quanti_caracter']= data['url'].apply(lambda i: letter_count(i))
data = trataOutlier(data)

data = data.dropna(subset=["Categoria"])
X = data.drop(["url","type","Categoria"], axis=1)
y = data["Categoria"]

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "features_lista.pkl")
print("\nLista de features salva como: features_lista.pkl\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nDistribuição original no treino:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nDistribuição após SMOTE:", Counter(y_train_res))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}
labels = ["benign", "defacement", "phishing", "malware"]

for nome, modelo in modelos.items():
    if nome in ["LogisticRegression", "SVM"]:
        modelo.fit(X_train_scaled, y_train_res)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train_res, y_train_res)
        y_pred = modelo.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if nome == "RandomForest":
        joblib.dump(modelo, "RandomForest_phishing.pkl")
        print("RandomForest salvo como: RandomForest_phishing.pkl")
