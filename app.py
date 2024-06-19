import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


df = pd.read_csv('ratings_Electronics.csv', header=None)
df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
df = df.drop('timestamp', axis=1)
df_copy = df.copy(deep=True)

counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])
final_ratings_matrix.set_index(['user_index'], inplace=True)

def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_score = [tup[1] for tup in similarity]
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])
    return most_similar_users, similarity_score

def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)])
    recommendations = []
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)])
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    return recommendations[:num_of_products]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_index = int(request.form['user_index'])
    num_of_products = int(request.form['num_of_products'])

    try:
        recs = recommendations(user_index, num_of_products, final_ratings_matrix)
        return render_template('index.html', recommendations=recs)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
