import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.*")
from flask import Flask, render_template, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
cred = credentials.Certificate('firebase.json')

firebase_admin.initialize_app(cred)
db = firestore.client()

user_interests_data = {
}

def fetch_all_documents():
    collection_ref = db.collection('users')
    docs = collection_ref.stream()
    for doc in docs:
        user_id = doc.id
        user_data = doc.to_dict()

        if 'interests' in user_data:
            interests_str = ', '.join(user_data['interests'])
            user_interests_data[user_data['name']] = interests_str
        else:
            # Handle the case where the 'interests' key does not exist
            print(f"User {user_data['name']} has no 'interests' field in Firestore.")
    

print("Fetching all documents")
fetch_all_documents()
print(user_interests_data )

local_model_path = os.path.join('local_models', 'all-MiniLM-L12-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def generate_average_embedding(sentences):
    embeddings = model.encode(sentences)
    return np.mean(embeddings, axis=0)

user_embeddings = {}
for user_id, interests_str in user_interests_data.items():
    interests = interests_str.split(', ')
    user_embedding = generate_average_embedding(interests)
    user_embeddings[user_id] = {
        'embedding': user_embedding,
        'interests': interests
    }

def find_k_nearest_neighbors(user_id, k=5):
    user_embedding = user_embeddings[user_id]['embedding']
    similarities = []
    for other_user_id, data in user_embeddings.items():
        if other_user_id != user_id:
            other_embedding = data['embedding']
            similarity = np.dot(user_embedding, other_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(other_embedding))
            similarities.append((other_user_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    nearest_neighbors = [(sim[0], user_embeddings[sim[0]]['interests']) for sim in similarities[:k]]
    return nearest_neighbors

def add_new_user(user_id, interests_str):
    global user_interests_data
    user_interests_data[user_id] = interests_str
    interests = interests_str.split(', ')
    user_embedding = generate_average_embedding(interests)
    user_embeddings[user_id] = {
        'embedding': user_embedding,
        'interests': interests
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_interests = request.form['user_interests']
        interests_list = [interest.strip() for interest in user_interests.split(',')]
        user_embedding = generate_average_embedding(interests_list)

        similarities = []
        for user_id, data in user_embeddings.items():
            similarity = np.dot(user_embedding, data['embedding']) / (np.linalg.norm(user_embedding) * np.linalg.norm(data['embedding']))
            similarities.append((user_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = [(sim[0], user_embeddings[sim[0]]['interests']) for sim in similarities[:5]]  

        new_user_id = f"user{len(user_interests_data) + 1}"
        add_new_user(new_user_id, user_interests)

        similar_users = [(user_id, interests) for user_id, interests in similar_users if user_id != new_user_id]

        return render_template('index.html', similar_users=similar_users)

    return render_template('index.html')

@app.route('/api/similar_users', methods=['POST'])
def get_similar_users():
    try:
        fetch_all_documents()
        print(user_interests_data )
        data = request.get_json()
        user_interests = data.get('user_interests', '')
        interests_list = [interest.strip() for interest in user_interests.split(',')]
        user_embedding = generate_average_embedding(interests_list)

        similarities = []
        for user_id, data in user_embeddings.items():
            similarity = np.dot(user_embedding, data['embedding']) / (np.linalg.norm(user_embedding) * np.linalg.norm(data['embedding']))
            similarities.append((user_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = [(sim[0], user_embeddings[sim[0]]['interests']) for sim in similarities[:5]]
        print(jsonify({'similar_users': similar_users}))
        return jsonify({'similar_users': similar_users})
    except Exception as e:
        print(jsonify({'error': str(e)}), 400)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)