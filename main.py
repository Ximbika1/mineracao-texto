import bz2          # para ler o .bz2
import re           # para limpar o texto com regex
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


'''
---------------------------|| Limpando os dados e fazendo a conversao para DataFrame ||--------------------------------------
'''

def clean_text(s: str) -> str:
    """
    Limpa o texto:
    - converte para minúsculas
    - remove URLs, @usuarios, hashtags
    - mantém apenas letras e espaços
    """    
    s = str(s).lower()  # tudo minúsculo
    # Remove URLs
    s = re.sub(r"http\S+|www\S+", " ", s)
    # Remove @usuario e #hashtag (se tiver)
    s = re.sub(r"(@\w+|#\w+)", " ", s)
    # Mantém só letras e espaços (incluindo acentos)
    s = re.sub(r"[^a-zà-úçñãõâêîôûäëïöü\s]", " ", s)
    # Troca múltiplos espaços por 1 só
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_fasttext_bz2(path, limit=None):
    """
    Lê o arquivo .bz2 no formato fastText e devolve
    um DataFrame com colunas ['text', 'sentiment'].
    """
    labels = []
    texts = []
    count = 0

    with bz2.open(path, mode="rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if limit is not None and count >= limit:
                break  # para de ler depois do limite

            line = line.strip()
            if not line:
                continue

            partes = line.split(" ", 1)
            if len(partes) < 2:
                continue

            label_raw = partes[0]      # ex: "__label__1"
            text_raw  = partes[1]      # resto da frase

            labels.append(label_raw)
            texts.append(text_raw)
            count += 1

    df = pd.DataFrame({"label_raw": labels, "text_raw": texts})

    # tira o prefixo "__label__"
    df["label"] = df["label_raw"].str.replace("__label__", "", regex=False)

    # mapeia para positivo/negativo
    df["sentiment"] = df["label"].map({"1": "negativo", "2": "positivo"})
    df = df.dropna(subset=["sentiment"]).copy()

    # limpa o texto
    df["text"] = df["text_raw"].apply(clean_text)

    return df[["text", "sentiment"]]

'''
---------------------------|| Funções de preparação e modelagem ||--------------------------------------
'''

def split_train_test(df, test_size=0.2, random_state=42):
    """
    Recebe o DataFrame com ['text', 'sentiment'] e devolve
    X_train, X_test, y_train, y_test.
    """
    X_text = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nTamanhos dos conjuntos:")
    print("Treino:", len(X_train))
    print("Teste:", len(X_test))

    return X_train, X_test, y_train, y_test


def vectorize_tfidf(X_train, X_test,
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=5):
    """
    Cria o vetorizador TF-IDF e transforma X_train e X_test.
    Retorna: vectorizer, X_train_tfidf, X_test_tfidf, feature_names
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names_out()

    print("\n=== TF-IDF: Informações da matriz ===")
    print("Shape X_train_tfidf (docs_treino x termos):", X_train_tfidf.shape)
    print("Shape X_test_tfidf  (docs_teste  x termos):", X_test_tfidf.shape)
    print("\nTamanho do vocabulário TF-IDF:", len(feature_names))
    print("Algumas palavras do vocabulário:", feature_names[:30])

    # Amostra da matriz TF-IDF (primeira linha)
    sample_row = X_train_tfidf[0].toarray()[0]
    print("\nPrimeira linha da matriz TF-IDF (primeiros 30 valores):")
    print(sample_row[:30])

    return vectorizer, X_train_tfidf, X_test_tfidf, feature_names


'''
---------------------------|| Regressão logística ||--------------------------------------
'''

def train_logistic_regression(X_train_tfidf, y_train):
    """
    Treina a Regressão Logística e devolve o modelo treinado.
    """
    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    log_reg.fit(X_train_tfidf, y_train)

    print("\nModelo de Regressão Logística treinado.")
    print("Classes aprendidas:", log_reg.classes_)

    return log_reg


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Avalia o modelo no conjunto de teste.
    Retorna: dict com métricas e a matriz de confusão em DataFrame.
    """
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Métricas de desempenho ===")
    print(f"Acurácia no conjunto de teste: {acc:.4f}\n")

    report = classification_report(y_test, y_pred, digits=4)
    print("Relatório de classificação:")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=["negativo", "positivo"])
    cm_df = pd.DataFrame(
        cm,
        index=["Real negativo", "Real positivo"],
        columns=["Predito negativo", "Predito positivo"]
    )

    print("\nMatriz de Confusão:")
    print(cm_df)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm_df
    }
    return metrics


'''
---------------------------|| Extração das palavras mais importantes ||--------------------------------------
'''

def get_top_words(model, feature_names, positive_class="positivo", top_n=20):
    """
    Retorna dois DataFrames:
      - df_top_pos: top palavras mais associadas à classe positiva
      - df_top_neg: top palavras mais associadas à classe negativa

    Trata corretamente o caso binário (coef_ com apenas 1 linha)
    e o caso multi-classe.
    """
    classes = model.classes_
    coefs_matrix = model.coef_
    n_classes = len(classes)

    print("\nOrdem das classes no modelo:", classes)
    print("Formato de coef_:", coefs_matrix.shape)

    # Caso binário: coef_.shape = (1, n_features)
    if n_classes == 2 and coefs_matrix.shape[0] == 1:
        # Pelo scikit-learn, coef_[0] está associado à classes_[1]
        if positive_class == classes[1]:
            coefs = coefs_matrix[0]
        elif positive_class == classes[0]:
            # Se alguém quiser usar a classe[0] como "positiva",
            # invertemos o sinal.
            coefs = -coefs_matrix[0]
        else:
            raise ValueError(f"positive_class '{positive_class}' não está em {classes}")
    else:
        # Caso geral (multi-classe): uma linha por classe
        idx_pos_arr = np.where(classes == positive_class)[0]
        if len(idx_pos_arr) == 0:
            raise ValueError(f"positive_class '{positive_class}' não está em {classes}")
        idx_pos = idx_pos_arr[0]
        coefs = coefs_matrix[idx_pos]

    # top N positivos
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_pos_words = [(feature_names[i], coefs[i]) for i in top_pos_idx]
    df_top_pos = pd.DataFrame(top_pos_words, columns=["palavra", "peso"])

    # top N negativos (menores pesos)
    top_neg_idx = np.argsort(coefs)[:top_n]
    top_neg_words = [(feature_names[i], coefs[i]) for i in top_neg_idx]
    df_top_neg = pd.DataFrame(top_neg_words, columns=["palavra", "peso"])

    print("\n=== Palavras mais associadas a avaliações POSITIVAS ===")
    print(df_top_pos)

    print("\n=== Palavras mais associadas a avaliações NEGATIVAS ===")
    print(df_top_neg)

    return df_top_pos, df_top_neg


'''
---------------------------|| Script principal: chamando as funções ||--------------------------------------
'''

if __name__ == "__main__":
    caminho_arquivo = "base/test.ft.txt.bz2" #Local onde a base de dados está localizado

    # 1) Carrega a base
    df = load_fasttext_bz2(caminho_arquivo, limit=20000)  # ou None pra usar tudo

    print("Amostra da base:")
    print(df.head(10))
    print("\nQuantidade total de exemplos:", len(df))
    print("\nDistribuição de classes:")
    print(df["sentiment"].value_counts())

    # 2) Separa treino/teste
    X_train, X_test, y_train, y_test = split_train_test(df)

    # 3) Vetoriza com TF-IDF
    vectorizer, X_train_tfidf, X_test_tfidf, feature_names = vectorize_tfidf(
        X_train, X_test,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=5
    )

    # 4) Treina a Regressão Logística
    model = train_logistic_regression(X_train_tfidf, y_train)

    # 5) Avalia o modelo
    metrics = evaluate_model(model, X_test_tfidf, y_test)

    # 6) Palavras mais importantes para classe positivo/negativo
    df_top_pos, df_top_neg = get_top_words(model, feature_names,
                                           positive_class="positivo",
                                           top_n=20)

    # Aqui você pode salvar tabelas em CSV se quiser usar no site:
    # df_top_pos.to_csv("download/top_palavras_positivas.csv", index=False)
    # df_top_neg.to_csv("download/top_palavras_negativas.csv", index=False)
    # metrics["confusion_matrix"].to_csv("download/matriz_confusao.csv")
    
    