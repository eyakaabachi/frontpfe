import csv

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from django.shortcuts import render
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dash import Input, Output, dcc, html
from .dashboarding import generate_dash_app


def home(request):
    return render(request, 'home.html')


def cancer(request):
    return render(request, 'predict.html')


def diabete(request):
    return render(request, 'diabete.html')


def heart(request):
    return render(request, 'heart.html')


def mental(request):
    return render(request, 'mental.html')


def login(request):
    return render(request, 'login.html')


def profile(request):
    return render(request, 'profile.html')


def get_recommended_doctors(patient_cluster, num_recommendations=5):
    # The given code to retrieve recommended doctors
    doc_df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\healthcare_docs.csv")
    doc_df.rename(columns={'CredentialType': 'Specialty'}, inplace=True)
    doc_df['FullName'] = doc_df['LastName'].str.cat(doc_df['FirstName'], sep=' ')
    doc_df.drop(["LastName", "FirstName"], axis=1, inplace=True)
    filtered_doc_df = doc_df.copy()

    for index, row in doc_df.iterrows():
        if ("Nurse" in row['Specialty']) or ("Pharmacy" in row['Specialty']) or ("Nursing" in row['Specialty']) or (
                "Dental" in row['Specialty']):
            filtered_doc_df.drop(index, inplace=True)

    filtered_doc_df.reset_index(drop=True, inplace=True)
    doc_df = filtered_doc_df

    specialties = doc_df['Specialty'].fillna('').astype(str).tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(specialties)

    patient_cluster_str = [str(patient_cluster)]
    patient_tfidf = tfidf_vectorizer.transform(patient_cluster_str)

    similarity_scores = cosine_similarity(patient_tfidf, tfidf_matrix)

    top_indices = similarity_scores.argsort()[0][-num_recommendations:][::-1]
    top_doctors = doc_df.iloc[top_indices]

    recommended_doctors = top_doctors.to_dict(orient='records')

    return recommended_doctors


def result_cancer(request):
    df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\new_datapfe.csv")
    features = ['SEX', 'EXERANY2', '_RFBING5', '_RFDRHV5', '_FRTLT1', '_VEGLT1',
                'BPMEDS', 'BLOODCHO', 'ADDEPEV2', 'TOLDHI2', 'CVDINFR4', 'CVDCRHD4',
                'CVDSTRK3', '_MICHD', 'DIABETE3', 'SMOKE100', 'SMOKDAY2', 'USENOW3',
                '_SMOKER3', '_RFSMOK3', 'CHCSCNCR']
    daf = df[features]
    df_std = (daf - daf.mean()) / daf.std()

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_std), columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmodel = kmeans.fit_predict(df_pca)

    X = daf.drop('CHCSCNCR', axis=1)
    y = daf['CHCSCNCR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3, 'iterations': 100}

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = float(request.GET['n11'])
    var12 = float(request.GET['n12'])
    var13 = float(request.GET['n13'])
    var14 = float(request.GET['n14'])
    var15 = float(request.GET['n15'])
    var16 = float(request.GET['n16'])
    var17 = float(request.GET['n17'])
    var18 = float(request.GET['n18'])
    var19 = float(request.GET['n19'])
    var20 = float(request.GET['n20'])

    catboost_md = model.predict(np.array([var1, var2, var3, var4, var5, var6,
                                          var7, var8, var9, var10, var11, var12,
                                          var13, var14, var15, var16, var17,
                                          var18, var19, var20]).reshape(1, -1))
    # .reshape(1, -1)
    pred = round(catboost_md[0])
    if pred == 0:
        res = "Quit smoking, limit alcohol consumption, and try to stay healthy to avoid getting cancer."
        recommended_doctors1 = None
    elif pred == 1:
        res = "Congratulations, the patient is healthy."
        recommended_doctors1 = None
    else:
        recommended_doctors1 = get_recommended_doctors(pred)
        res = "Patient is at risk! Please visit one of these specialists:"

    return render(request, 'predict.html', {'result1': res, 'recommended_doctors1': recommended_doctors1})


def dashboard(request):
    return render(request, 'index.html')


def result_diabete(request):
    df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\new_datapfe.csv")
    features = ['GENHLTH', 'EXERANY2', '_RFBING5', '_RFDRHV5', 'DRNKANY5', '_VEGLT1',
                'BPHIGH4', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDINFR4',
                'CVDCRHD4', 'CVDSTRK3', '_MICHD', 'CHCSCNCR', 'SMOKDAY2', 'USENOW3',
                '_SMOKER3', '_RFSMOK3', 'DIABETE3']
    daf = df[features]
    # Standardize the data
    df_std = (daf - daf.mean()) / daf.std()

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_std), columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmodel = kmeans.fit_predict(df_pca)
    # Split data into features (X) and target variable (y)
    X = daf.drop('DIABETE3', axis=1)
    y = daf['DIABETE3']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the hyperparameters for the CatBoost algorithm
    params = {'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3, 'iterations': 100}

    # Initialize the CatBoostClassifier object
    # with the defined hyperparameters and fit it on the training set
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = float(request.GET['n11'])
    var12 = float(request.GET['n12'])
    var13 = float(request.GET['n13'])
    var14 = float(request.GET['n14'])
    var15 = float(request.GET['n15'])
    var16 = float(request.GET['n16'])
    var17 = float(request.GET['n17'])
    var18 = float(request.GET['n18'])
    var19 = float(request.GET['n19'])
    var20 = float(request.GET['n20'])

    catboost_md = model.predict(np.array([var1, var2, var3, var4, var5, var6,
                                          var7, var8, var9, var10, var11, var12,
                                          var13, var14, var15, var16, var17,
                                          var18, var19, var20]).reshape(1, -1))
    pred = np.round(catboost_md[0], decimals=2)
    if pred == 0:
        res = "Congratulations, the patient is healthy."
        recommended_doctors2 = None
    elif pred == 1:
        res = "Quit smoking, limit alcohol consumption and try to stay healthy to avoid getting diabetes."
        recommended_doctors2 = None
    else:  # Check if there is a predicted risk (assuming cluster 2 represents the high-risk cluster)
        recommended_doctors2 = get_recommended_doctors(pred)
        res = "Patient is at risk! Please visit one of these specialists:"

    return render(request, 'diabete.html', {'result2': res, 'recommended_doctors2': recommended_doctors2})


def result_heart(request):
    df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\new_datapfe.csv")
    features = ['SEX', 'EXERANY2', '_RFDRHV5', '_FRTLT1', '_VEGLT1', 'BPHIGH4',
                'BPMEDS', 'BLOODCHO', 'ADDEPEV2', 'TOLDHI2', 'CVDINFR4', 'CVDSTRK3',
                '_MICHD', 'CHCSCNCR', 'DIABETE3', 'SMOKE100', 'SMOKDAY2', 'USENOW3',
                '_SMOKER3', '_RFSMOK3', 'CVDCRHD4']
    daf = df[features]
    # Standardize the data
    df_std = (daf - daf.mean()) / daf.std()

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_std), columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmodel = kmeans.fit_predict(df_pca)

    # Split data into features (X) and target variable (y)
    X = daf.drop('CVDCRHD4', axis=1)
    y = daf['CVDCRHD4']
    # Split data into training and testing sets
    cluster_labels = kmeans.labels_
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    params = {'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3, 'iterations': 100}


    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = float(request.GET['n11'])
    var12 = float(request.GET['n12'])
    var13 = float(request.GET['n13'])
    var14 = float(request.GET['n14'])
    var15 = float(request.GET['n15'])
    var16 = float(request.GET['n16'])
    var17 = float(request.GET['n17'])
    var18 = float(request.GET['n18'])
    var19 = float(request.GET['n19'])
    var20 = float(request.GET['n20'])

    catboost_md = model.predict(np.array([var1, var2, var3, var4, var5, var6,
                                          var7, var8, var9, var10, var11, var12,
                                          var13, var14, var15, var16, var17,
                                          var18, var19, var20]).reshape(1, -1))
    # .reshape(1, -1)
    pred = round(catboost_md[0])
    if pred == 0:
        res = "Quit smoking, limit alcohol consumption and try to stay healthy to avoid getting any type of heart disease."
        recommended_doctors4 = None
    elif pred == 1:
        res = "Congratulations, the patient is healthy."
        recommended_doctors4 = None
    else:  # Check if there is a predicted risk (assuming cluster 2 represents the high-risk cluster)
        recommended_doctors4 = get_recommended_doctors(pred)
        res = "Patient is at risk! Please visit one of these specialists:"

    return render(request, 'heart.html', {'result4': res, 'recommended_doctors4': recommended_doctors4})


def result_mental(request):
    df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\new_datapfe.csv")
    features = ['SEX', 'EXERANY2', '_RFDRHV5', 'MENTHLTH', '_FRTLT1', '_VEGLT1',
                'BPMEDS', 'BLOODCHO', 'TOLDHI2', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3',
                '_MICHD', 'CHCSCNCR', 'DIABETE3', 'SMOKE100', 'SMOKDAY2', 'USENOW3', 'ADDEPEV2',
                '_SMOKER3', '_RFSMOK3']
    daf = df[features]
    # Standardize the data
    df_std = (daf - daf.mean()) / daf.std()

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_std), columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmodel = kmeans.fit_predict(df_pca)

    # Split data into features (X) and target variable (y)
    X = daf.drop('ADDEPEV2', axis=1)
    y = daf['ADDEPEV2']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the hyperparameters for the CatBoost algorithm
    params = {'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3, 'iterations': 100}

    # Initialize the CatBoostClassifier object
    # with the defined hyperparameters and fit it on the training set
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])
    var11 = float(request.GET['n11'])
    var12 = float(request.GET['n12'])
    var13 = float(request.GET['n13'])
    var14 = float(request.GET['n14'])
    var15 = float(request.GET['n15'])
    var16 = float(request.GET['n16'])
    var17 = float(request.GET['n17'])
    var18 = float(request.GET['n18'])
    var19 = float(request.GET['n19'])
    var20 = float(request.GET['n20'])

    catboost_md = model.predict(np.array([var1, var2, var3, var4, var5, var6,
                                          var7, var8, var9, var10, var11, var12,
                                          var13, var14, var15, var16, var17,
                                          var18, var19, var20]).reshape(1, -1))
    # .reshape(1, -1)
    pred = round(catboost_md[0])
    if pred == 0:
        res = "Quit smoking, limit alcohol consumption and try to stay healthy for a good mental health."
        recommended_doctors3 = None
    elif pred == 1:
        res = "Congratulations, the patient is healthy."
        recommended_doctors3 = None
    else:  # Check if there is a predicted risk (assuming cluster 2 represents the high-risk cluster)
        recommended_doctors3 = get_recommended_doctors(pred)
        res = "Patient is at risk! Please visit one of these specialists:"

    return render(request, 'mental.html', {'result3': res, 'recommended_doctors3': recommended_doctors3})


def patients(request):
    csv_file_path = "C://Users//Eya Kaabachi//Desktop//new_datapfe.csv"

    # Read the CSV file and store data in a list of dictionaries
    df = pd.read_csv(csv_file_path)
    df['_BMI5'] = df['_BMI5'].round(3)
    # Rename the columns
    df = df.rename(columns={'_AGEG5YR': 'Age', '_BMI5': 'BMI'})

    data = df.head(1000).to_dict(orient='records')

    return render(request, "patients.html", {'data': data})


def doctors(request):
    csv_file_path = "C://Users//Eya Kaabachi//Desktop//healthcare_docs.csv"

    # Read the CSV file and store data in a list of dictionaries
    df = pd.read_csv(csv_file_path)
    # Rename the columns
    data = df.to_dict(orient='records')
    return render(request, "doctors.html", {'data': data})
