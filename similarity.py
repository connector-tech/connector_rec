#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gensim
import json
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def process_and_cluster(json_file, num_clusters=3):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Data is not a list of dictionaries. DataFrame creation not applicable.")
    
        df = pd.DataFrame(data)
        df.insert(0, 'id', range(1, len(df) + 1))
        df.drop(columns=['Religion', 'Activities', 'Favorite sport', 'Dream job', 'Interests'], inplace=True)
    
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data. {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None

    def preprocess_df(df):
        df_text = df.drop(columns=['id'])
        df_text = df_text.apply(lambda col: col.astype(str).str.lower())
        df_text = df_text.apply(lambda col: col.str.replace(r'[^a-zA-Z\s]', '', regex=True))
        df_text = df_text.apply(lambda col: col.str.replace(r'\s+', ' ', regex=True))
        df_text = df_text.apply(lambda col: col.apply(nltk.word_tokenize))

        stop_words = set(stopwords.words('english'))
        df_text = df_text.apply(lambda col: col.apply(lambda tokens: [token for token in tokens if token not in stop_words]))

        lemmatizer = WordNetLemmatizer()
        df_text = df_text.apply(lambda col: col.apply(lambda tokens: [lemmatizer.lemmatize(token, pos="v") for token in tokens]))
        df_text = df_text.apply(lambda col: col.apply(' '.join))

        preprocessed_df = pd.concat([df[['id']], df_text], axis=1)
        return preprocessed_df

    preprocessed_df = preprocess_df(df)

    def cluster_users_with_cosine_and_kmeans(data_df, parameter_columns, num_clusters=3):
        parameter_weights = [1.0] * len(parameter_columns)

        def combine_weighted_features(row, weights):
            combined_features = []
            for i, value in enumerate(row[parameter_columns]):
                value = str(value)
                weighted_feature = value + ' ' * int(weights[i] * len(value))
                combined_features.append(weighted_feature)
            return ' '.join(combined_features)

        data_df['weighted_features'] = data_df.apply(combine_weighted_features, axis=1, args=[parameter_weights])
        vectorizer = TfidfVectorizer()
        combined_feature_vectors = vectorizer.fit_transform(data_df['weighted_features'])
        
        cosine_sim_matrix = cosine_similarity(combined_feature_vectors)
        
        scaler = MaxAbsScaler()
        scaled_cosine_sim_matrix = scaler.fit_transform(cosine_sim_matrix)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_cosine_sim_matrix)
        
        similar_users = []
        num_users = len(data_df)
        for i in range(num_users):
            for j in range(i+1, num_users):
                similarity = cosine_sim_matrix[i, j]
                user1_id, user2_id = data_df.loc[i, 'id'], data_df.loc[j, 'id']
                if (user1_id, user2_id) not in similar_users and (user2_id, user1_id) not in similar_users:
                    similar_users.append((user1_id, user2_id, similarity))
        
        return similar_users, cosine_sim_matrix, cluster_labels

    parameter_columns = list(preprocessed_df.columns[1:])
    similar_users, cosine_sim_matrix, cluster_labels = cluster_users_with_cosine_and_kmeans(preprocessed_df, parameter_columns, num_clusters)
    df_sim = pd.DataFrame(similar_users, columns=['id', 'id2', 'sim'])

    # Преобразование DataFrame в JSON
    df_sim_json = df_sim.to_json(orient='records')

    return df_sim_json


# Пример использования функции
df_sim_json = process_and_cluster('syntetic_large_en.json', num_clusters=3)
#print(df_sim_json)


# In[ ]:




