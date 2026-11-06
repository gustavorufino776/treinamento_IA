#IMPORTANTE: antes de começar, instale os pacotes contidos em requeriments.txt
#
#python -m pip install -r requeriments.txt
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# importante: Instruções para Repositório de dados
# 1. Criar um documento no google sheet
# 2. Insira dados com estes cabeçalhos: faltas, nota_1, nota_2, resutado 
# 3. Adicione de 10 a 15 linhas conforme o exemplo abaixo:
#   8,2.2,4.5,reprovado
# 4. Ir em "arquivo" -> "Compartilhar" -> "publicar na web"
# 5. Selecionar "Valores separados por vírgula (.csv)" e clicar em "Publicar"
# 6. copiar link gerado e colar abaixo

URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSYyDl15VH0k6VXX6daTw-LxBwId5EVEhGUBc5qz5mmH9ECuQM-j3QLcV8BZGXglkIh8wUo9e7RNh5h/pub?output=csv"

ARQUIVO_MODELO = 'modelo_desempenho.pkl'

def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS} ...")
    data = pd.read_csv(URL_DADOS)
    print("--Dados carregados--")
    print(data.head())

    print("--- Preparando dados para o treino---")
    features = ["faltas","nota_1", "nota_2"]
    target = "resutado"
    X = data[features]
    Y = data[target]

    modelo = KNeighborsClassifier(n_neighbors=3)

    modelo.fit(X, Y)
    print(f"--- Modelo Treinado! Classes: {modelo.classes_} ---")

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"--- modelo salvo com sucesso em '{ARQUIVO_MODELO} ---")

if __name__ == "__main__":
    treinar_modelo()