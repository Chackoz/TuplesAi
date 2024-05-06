from flask import Flask, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Initialize user interests data (replace this with your actual data)
user_interests_data = {
    'user1': "web development, reading novels, singing",
    'user2': "cooking, watching movies, anime",
    'user3': "Design, football, playing guitar",
    'user4': "Coding, dance, music"
}

# Function to generate embeddings for a list of sentences and average them
def generate_average_embedding(sentences):
    embeddings = model.encode(sentences)
    return np.mean(embeddings, axis=0)

# Function to find the k nearest neighbors for a given user
def find_k_nearest_neighbors(user_embedding, k=5):
    similarities = []
    for user_id, data in user_interests_data.items():
        other_embedding = generate_average_embedding(data.split(', '))
        similarity = np.dot(user_embedding, other_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(other_embedding))
        similarities.append((user_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    nearest_neighbors = [(sim[0], user_interests_data[sim[0]].split(', ')) for sim in similarities[:k]]
    return nearest_neighbors

@app.route('/api/similar_users', methods=['POST'])
def get_similar_users():
    try:
        data = request.get_json()
        user_interests = data.get('user_interests', '')
        interests_list = [interest.strip() for interest in user_interests.split(',')]
        user_embedding = generate_average_embedding(interests_list)

        similar_users = find_k_nearest_neighbors(user_embedding)

        return jsonify({'similar_users': similar_users})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
