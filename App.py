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

 
# **2. Preprocessing Data**
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

# **3. Feature Extraction dan Algoritma**
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

# **4. Evaluasi Algoritma**
with st.expander("Evaluasi Algoritma"):
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

    st.markdown("""
        **Output:**
        1. Skenario 1: TF-IDF + Naive Bayes → Akurasi dan evaluasi performa ditampilkan.
        2. Skenario 2: TF-IDF + SVM → Akurasi dan evaluasi performa ditampilkan.
        3. Skenario 3: Bag-of-Words + Naive Bayes → Akurasi dan evaluasi performa ditampilkan.
        4. Skenario 4: Bag-of-Words + SVM → Akurasi dan evaluasi performa ditampilkan.
        """)

# **5. Skenario Eksperimen**
with st.expander("Skenario Eksperimen"):
    # 4 Skenario
    # Skenario 1: TF-IDF dan Naive Bayes
    # Skenario 2: Menggunakan TD-IDF dan SVM
    # Skenario 3: Menggunakan Bag-of-Words dan Naive Bayes.
    # Skenario 4: Menggunakan Bag-of-Words dan SVM
    
    st.subheader("Skenario 1: Menggunakan TF-IDF dan Naive Bayes.")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb_tfidf = nb.predict(X_test_tfidf)
    report_nb_tfidf = classification_report(y_test, y_pred_nb_tfidf, target_names=le.classes_, output_dict=True)
    df_nb_tfidf = pd.DataFrame(report_nb_tfidf).transpose()
    st.write("Hasil Evaluasi:")
    st.dataframe(df_nb_tfidf)
    st.markdown("""
    **Alasan:**
    - TF-IDF menangkap bobot penting kata-kata unik dalam ulasan.
    - Naive Bayes bekerja dengan baik untuk data teks dan menghasilkan hasil yang cepat.
    - Namun, Precision sedikit lebih rendah dibandingkan SVM karena asumsi independensi fitur.
    """)

    st.subheader("Skenario 2: Menggunakan TF-IDF dan SVM")
    svm = SVC(kernel='linear')
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm_tfidf = svm.predict(X_test_tfidf)
    report_svm_tfidf = classification_report(y_test, y_pred_svm_tfidf, target_names=le.classes_, output_dict=True)
    df_svm_tfidf = pd.DataFrame(report_svm_tfidf).transpose()
    st.write("Hasil Evaluasi:")
    st.dataframe(df_svm_tfidf)
    st.markdown("""
    **Alasan:**
    - SVM dengan kernel linear cocok untuk data yang terpisah secara linear.
    - TF-IDF membantu mengurangi noise dari kata-kata umum sehingga performa SVM meningkat.
    - SVM menghasilkan Precision lebih tinggi dibandingkan Naive Bayes karena keunggulan dalam menangani fitur teks kompleks.
    """)

    st.subheader("Skenario 3: Menggunakan Bag-of-Words dan Naive Bayes")
    bow_vectorizer = CountVectorizer(max_features=500)
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_bow, y_train)
    y_pred_nb_bow = nb_bow.predict(X_test_bow)
    report_nb_bow = classification_report(y_test, y_pred_nb_bow, target_names=le.classes_, output_dict=True)
    df_nb_bow = pd.DataFrame(report_nb_bow).transpose()
    st.write("Hasil Evaluasi:")
    st.dataframe(df_nb_bow)
    st.markdown("""
    **Alasan:**
    - Bag-of-Words bekerja dengan menghitung frekuensi kata tanpa memperhatikan bobotnya.
    - Naive Bayes cocok untuk data yang lebih sederhana, tetapi performanya lebih rendah karena informasi kontekstual dari kata diabaikan.
    """)

    st.subheader("Skenario 4: Menggunakan Bag-of-Words dan SVM")
    svm_bow = SVC(kernel='linear')
    svm_bow.fit(X_train_bow, y_train)
    y_pred_svm_bow = svm_bow.predict(X_test_bow)
    report_svm_bow = classification_report(y_test, y_pred_svm_bow, target_names=le.classes_, output_dict=True)
    df_svm_bow = pd.DataFrame(report_svm_bow).transpose()
    st.write("Hasil Evaluasi:")
    st.dataframe(df_svm_bow)
    st.markdown("""
    **Alasan:**
    - SVM tetap menghasilkan hasil yang baik meskipun menggunakan Bag-of-Words.
    - Namun, hasilnya sedikit lebih rendah dibandingkan TF-IDF karena Bag-of-Words tidak memberikan bobot pada kata penting.
    """)
    
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

    st.markdown("""
        **Confusion Matrix - Naive Bayes:**
        - True Negatives (TN): 97 ulasan negatif terklasifikasi dengan benar.
        - True Positives (TP): 64 ulasan positif terklasifikasi dengan benar.
        - False Positives (FP): 12 ulasan negatif salah diklasifikasikan sebagai positif.
        - False Negatives (FN): 27 ulasan positif salah diklasifikasikan sebagai negatif.

        **Confusion Matrix - SVM:**
        - True Negatives (TN): 92 ulasan negatif terklasifikasi dengan benar.
        - True Positives (TP): 65 ulasan positif terklasifikasi dengan benar.
        - False Positives (FP): 17 ulasan negatif salah diklasifikasikan sebagai positif.
        - False Negatives (FN): 26 ulasan positif salah diklasifikasikan sebagai negatif.
        
        **Perbandingan Naive Bayes vs SVM:**
        Naive Bayes memiliki jumlah True Negatives (97) yang lebih tinggi dibandingkan SVM (92), menunjukkan kemampuan lebih baik dalam mendeteksi ulasan negatif. SVM sedikit lebih unggul dalam mendeteksi ulasan positif dengan jumlah True Positives (65) dibandingkan Naive Bayes (64). False Positives dan False Negatives dari Naive Bayes lebih rendah pada ulasan negatif, tetapi SVM menunjukkan keseimbangan antara positif dan negatif.
        """)

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
    


# Visualisasi perbandingan hasil
    st.subheader("Perbandingan Performa Model")
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    # Data dari semua skenario
    skenario_scores = {
        "Skenario 1 (TF-IDF + Naive Bayes)": [
            df_nb_tfidf.loc['positif', 'precision'],
            df_nb_tfidf.loc['positif', 'recall'],
            df_nb_tfidf.loc['positif', 'f1-score']
        ],
        "Skenario 2 (TF-IDF + SVM)": [
            df_svm_tfidf.loc['positif', 'precision'],
            df_svm_tfidf.loc['positif', 'recall'],
            df_svm_tfidf.loc['positif', 'f1-score']
        ],
        "Skenario 3 (Bag-of-Words + Naive Bayes)": [
            df_nb_bow.loc['positif', 'precision'],
            df_nb_bow.loc['positif', 'recall'],
            df_nb_bow.loc['positif', 'f1-score']
        ],
        "Skenario 4 (Bag-of-Words + SVM)": [
            df_svm_bow.loc['positif', 'precision'],
            df_svm_bow.loc['positif', 'recall'],
            df_svm_bow.loc['positif', 'f1-score']
        ],
    }

    # Plot perbandingan
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (skenario, scores) in enumerate(skenario_scores.items()):
        ax.bar(x + i * width, scores, width, label=skenario)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Perbandingan Performa Model (Positif)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()

    st.pyplot(fig)
    st.markdown("""
        **Perbandingan Visualisasi Performa Model:**
        1. X-Axis (Sumbu X): Menunjukkan metrik evaluasi:
        - Precision: Keakuratan prediksi positif.
        - Recall: Kemampuan mendeteksi ulasan positif.
        - F1-Score: Kombinasi dari Precision dan Recall.
        2. Y-Axis (Sumbu Y): Menunjukkan nilai skor (0.0 - 1.0).
        3. Bar Chart:
        - Setiap skenario direpresentasikan oleh satu bar pada setiap metrik.
        - Warna atau posisi bar sesuai dengan skenario (Skenario 1, Skenario 2, dst.).
        4. Analisis Perbadingan:
        - Untuk setiap metrik (Precision, Recall, F1-Score), lihat bar mana yang paling tinggi.
        - Precision: Skenario 1 (TF-IDF + Naive Bayes) memiliki skor Precision tertinggi.
        - Recall: Semua model memiliki performa yang mirip, tetapi Skenario 2 (TF-IDF + SVM) sedikit unggul.
        - F1-Score: Nilai F1-Score konsisten dengan Recall; TF-IDF + SVM sedikit unggul.

        """)
    
    st.markdown("""
        **Kesimpulan:**
        1. TF-IDF dengan SVM (Skenario 2) menunjukkan performa terbaik secara keseluruhan karena Precision, Recall, dan F1-Score lebih tinggi atau stabil dibandingkan skenario lainnya.
        2. Bag-of-Words cenderung lebih rendah performanya dibandingkan TF-IDF, karena tidak memperhatikan bobot kata-kata unik.
        """)
# st.write("Selesai.")
