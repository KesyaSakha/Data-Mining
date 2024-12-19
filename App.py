# -*- coding: utf-8 -*-
"""Data Mining - Final Project"""

# Install google-play-scraper terlebih dahulu
# Run this in your terminal or separate cell:
# !pip install google-play-scraper
# !pip install streamlit

from google_play_scraper import Sort, reviews
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

# **1. Scraping Data Ulasan**
st.title("Analisis Ulasan ShopeePay")

with st.expander("Scraping Data Ulasan"):
    # Scraping data ulasan
    result, _ = reviews(
        'com.shopeepay.id',
        lang='id',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=1000
    )

    # Konversi ke DataFrame
    df = pd.DataFrame(result)
    df = df[['content', 'score']]
    df['label'] = df['score'].apply(lambda x: 'positif' if x > 3 else 'negatif')
    df.to_csv('ulasan_shopeepay.csv', index=False)
    st.write("Data berhasil disimpan.")
    # Tambahkan tombol unduh untuk file CSV
    st.download_button(
        label="Unduh File CSV",
        data=df.to_csv(index=False),
        file_name='ulasan_shopeepay.csv',
        mime='text/csv'
    )

 
# **3. Preprocessing Data**
with st.expander("Preprocessing Data"):
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter selain huruf
        text = text.lower()                      # Konversi ke huruf kecil
        return text

    df['clean_content'] = df['content'].apply(clean_text)

    # Encode label
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_content'], df['label_encoded'], test_size=0.2, random_state=42
    )

    st.write("Preprocessing selesai.")

# **4. Feature Extraction dan Algoritma**
with st.expander("Feature Extraction dan Algoritma"):
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model 1: Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)

    # Model 2: Support Vector Machine
    svm = SVC(kernel='linear')
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm = svm.predict(X_test_tfidf)

    st.write("Model berhasil dilatih.")


    # Assuming 'df' is your DataFrame and 'y_pred_nb' and 'y_pred_svm' are your predictions
    # Calculate the number of positive and negative reviews for each model
    nb_positif = sum(y_pred_nb)
    nb_negatif = len(y_pred_nb) - nb_positif
    svm_positif = sum(y_pred_svm)
    svm_negatif = len(y_pred_svm) - svm_positif

    # Evaluasi
    print("Evaluasi Naive Bayes:")
    print(classification_report(y_test, y_pred_nb, target_names=le.classes_))

    print("Evaluasi SVM:")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# **5. Evaluasi Model**
with st.expander("Evaluasi Model"):
    # Hasil evaluasi Naive Bayes
    report_nb = classification_report(y_test, y_pred_nb, target_names=['negatif', 'positif'], output_dict=True)
    df_nb = pd.DataFrame(report_nb).transpose()

    st.subheader("Evaluasi Naive Bayes")
    st.dataframe(df_nb)

    # Hasil evaluasi SVM
    report_svm = classification_report(y_test, y_pred_svm, target_names=['negatif', 'positif'], output_dict=True)
    df_svm = pd.DataFrame(report_svm).transpose()

    st.subheader("Evaluasi SVM")
    st.dataframe(df_svm)

# **6. Visualisasi Data**
with st.expander("Visualisasi Data"):
    st.subheader("Confusion Matrix")

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['Negatif', 'Positif'])
    disp_nb.plot(cmap='Blues')
    plt.title('Confusion Matrix - Naive Bayes')
    st.pyplot(plt)

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Negatif', 'Positif'])
    disp_svm.plot(cmap='Blues')
    plt.title('Confusion Matrix - SVM')
    st.pyplot(plt)

    st.subheader("Perbandingan Akurasi Model")

    metrics = ['Precision', 'Recall', 'F1-Score']
    nb_scores = [df_nb.loc['positif', 'precision'], df_nb.loc['positif', 'recall'], df_nb.loc['positif', 'f1-score']]  # Ambil nilai dari DataFrame
    svm_scores = [df_svm.loc['positif', 'precision'], df_svm.loc['positif', 'recall'], df_svm.loc['positif', 'f1-score']]  # Ambil nilai dari DataFrame

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, nb_scores, width, label='Naive Bayes')
    rects2 = ax.bar(x + width / 2, svm_scores, width, label='SVM')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Perbandingan Performa Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    st.pyplot(fig)

st.write("Selesai.")

with st.expander("Skenario Eksperimen"):
    st.subheader("Skenario 1: Menggunakan TF-IDF dan Naive Bayes.")
    st.subheader("Skenario 2: Menggunakan TF-IDF dan SVM")
    st.subheader("Skenario 3: Menggunakan Bag-of-Words dan Naive Bayes")
    st.subheader("Skenario 4: Menggunakan Bag-of-Words dan SVM")
