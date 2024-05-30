import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
import nltk
from nltk.stem import WordNetLemmatizer


def preprocess_df(df):
    df_text = df.drop(columns=['id'])
    df_text = df_text.apply(lambda col: col.astype(str).str.lower())
    df_text = df_text.apply(lambda col: col.str.replace(r'[^a-zA-Z\s]', '', regex=True))
    df_text = df_text.apply(lambda col: col.str.replace(r'\s+', ' ', regex=True))
    df_text = df_text.apply(lambda col: col.apply(nltk.word_tokenize))
    lemmatizer = WordNetLemmatizer()
    df_text = df_text.apply(
        lambda col: col.apply(lambda tokens: [lemmatizer.lemmatize(token, pos="v") for token in tokens]))
    df_text = df_text.apply(lambda col: col.apply(' '.join))
    preprocessed_df = pd.concat([df[['id']], df_text], axis=1)
    return preprocessed_df


def cluster_users_with_cosine_and_kmeans(data_df, parameter_columns, num_clusters=2):
    parameter_weights = [1.0] * len(parameter_columns)

    def combine_weighted_features(row, weights):
        combined_features = []
        for i, value in enumerate(row[parameter_columns]):
            value = str(value)
            weighted_feature = value + ' ' * int(weights[i] * len(value))
            combined_features.append(weighted_feature)
        return ' '.join(combined_features)

    data_df['weighted_features'] = data_df.apply(combine_weighted_features, axis=1, args=[parameter_weights])
    vectorizer = TfidfVectorizer(min_df=1)
    combined_feature_vectors = vectorizer.fit_transform(data_df['weighted_features'])
    cosine_sim_matrix = cosine_similarity(combined_feature_vectors)
    scaler = MaxAbsScaler()
    scaled_cosine_sim_matrix = scaler.fit_transform(cosine_sim_matrix)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_cosine_sim_matrix)
    similar_users = []
    num_users = len(data_df)
    for i in range(num_users):
        for j in range(i + 1, num_users):
            similarity = cosine_sim_matrix[i, j]
            user1_id, user2_id = data_df.loc[i, 'id'], data_df.loc[j, 'id']
            if (user1_id, user2_id) not in similar_users and (user2_id, user1_id) not in similar_users:
                similar_users.append((user1_id, user2_id, similarity))
    return similar_users, cosine_sim_matrix, cluster_labels


def process_and_train(profile_json, df_act_json, df_sim_json, num_clusters=2):
    profile_data = json.loads(profile_json)

    df_list = []
    for i, profile in enumerate(profile_data):
        profile_df = pd.DataFrame(profile, index=[i])
        profile_df.insert(0, 'id', i + 1)
        df_list.append(profile_df)
    df = pd.concat(df_list, ignore_index=True)

    act_data = json.loads(df_act_json)
    df_act = pd.DataFrame(act_data)
    for key in df_act.columns:
        df_act[key] = df_act[key].astype(int)

    preprocessed_df = preprocess_df(df)

    sim_data = json.loads(df_sim_json)
    df_sim = pd.DataFrame(sim_data)
    for key in ['id', 'id2']:
        df_sim[key] = df_sim[key].astype(int)
    df_sim['sim'] = df_sim['sim'].astype(float)

    parameter_columns = list(preprocessed_df.columns[1:])
    similar_users, cosine_sim_matrix, cluster_labels = cluster_users_with_cosine_and_kmeans(preprocessed_df,
                                                                                            parameter_columns,
                                                                                            num_clusters)

    df_act['Target_ID'] = df_act['Target_ID'].astype(int)
    preprocessed_df['id'] = preprocessed_df['id'].astype(int)
    df_act['Author_ID'] = df_act['Author_ID'].astype(int)
    df_act['Ratio_Duration_Messages'] = 1 / (df_act['Duration_Minutes'] / df_act['Num_Messages'])
    merged_df = pd.merge(df_act, df_sim, left_on=['Author_ID', 'Target_ID'], right_on=['id', 'id2'], how='inner')
    final_df = pd.merge(preprocessed_df, merged_df, left_on='id', right_on='Author_ID', how='inner')
    final_df = pd.merge(final_df, preprocessed_df, left_on='Target_ID', right_on='id', how='inner')
    author_ids = final_df['Author_ID']
    target_ids = final_df['Target_ID']
    final_df.drop(columns=['Author_ID', 'Target_ID', 'id_y', 'id2'], inplace=True)
    final_df.rename(columns={'id_x': 'Author_ID'}, inplace=True)
    final_df.drop(columns=['Author_ID', 'id'], inplace=True)

    vectorizers = {}
    X_text_features = []
    text_columns = ['Goals_x', 'Personality traits_x', 'Dreams and goals_x',
                    'Thoughts on life_x', 'Expectations_x', 'weighted_features_x',
                    'Goals_y', 'Personality traits_y', 'Dreams and goals_y',
                    'Thoughts on life_y', 'Expectations_y', 'weighted_features_y']

    for col in text_columns:
        vectorizer = TfidfVectorizer()
        X_text_feature = vectorizer.fit_transform(final_df[col])
        vectorizers[col] = vectorizer
        X_text_features.append(X_text_feature.toarray())

    X_text_features = np.concatenate(X_text_features, axis=1)
    X_numerical = final_df.drop(columns=text_columns).values
    X = np.concatenate((X_text_features, X_numerical), axis=1)
    y = final_df['Ratio_Duration_Messages'].values
    scaler1 = MinMaxScaler()
    y = scaler1.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test, author_ids_train, author_ids_test, target_ids_train, target_ids_test = train_test_split(
        X, y, author_ids, target_ids, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    predictions = model.predict(X_test)
    print(predictions)

    results_df = pd.DataFrame({
        'Author_ID': author_ids_test,
        'Target_ID': target_ids_test,
        'Prediction': predictions.flatten()
    })

    results_json = results_df.to_json(orient='records')
    return results_json


profile_json = json.dumps([
    {
        "Goals": "Christian gay rasizms you bro somersimte saddnes sun time clock reading sunset",
        "Personality traits": "Christian gay rasizms you bro somersimte saddnes sun time clock reading sunset",
        "Dreams and goals": "Christian gay rasizms you bro somersimte saddnes sun time clock reading sunset",
        "Thoughts on life": "Christian gay rasizms you bro somersimte saddnes sun time clock reading sunset",
        "Expectations": "Christian gay rasizms you bro somersimte saddnes sun time clock reading sunset"
    },
    {
        "Goals": "Another line of goals",
        "Personality traits": "Another line of personality traits",
        "Dreams and goals": "Another line of dreams and goals",
        "Thoughts on life": "Another line of thoughts on life",
        "Expectations": "Another line of expectations"
    },
    {
        "Goals": "More goals for testing",
        "Personality traits": "More personality traits for testing",
        "Dreams and goals": "More dreams and goals for testing",
        "Thoughts on life": "More thoughts on life for testing",
        "Expectations": "More expectations for testing"
    }
])

df_act = json.dumps({
    "Author_ID": ["1", "2", "3"],
    "Target_ID": ["2", "3", "1"],
    "Num_Messages": ["10", "15", "8"],
    "Duration_Minutes": ["30", "45", "25"]
})

df_sim = json.dumps({
    "id": ["1", "2", "3"],
    "id2": ["2", "3", "1"],
    "sim": ["0.8", "0.85", "0.75"]
})

# results_json = process_and_train(profile_json, df_act, df_sim, num_clusters=1)
