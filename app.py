import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.*")

import os
import json
import logging
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(__name__)
CORS(app)

# Firebase and Firestore configuration
try:
    firebase_cred_json = os.environ.get('FIREBASE_CREDENTIALS')
    
    if firebase_cred_json:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(json.loads(firebase_cred_json), temp_file)
            temp_file_path = temp_file.name

        try:
            cred = credentials.Certificate(temp_file_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization error: {e}")
        finally:
            os.unlink(temp_file_path)  # Remove temporary file
    else:
        logger.error("No Firebase credentials provided")
        db = None

except Exception as e:
    logger.critical(f"Firebase configuration error: {e}")
    db = None

# Global variables with type hints
user_interests_data: Dict[str, Dict] = {}
user_embeddings: Dict[str, Dict] = {}

# Fetch all documents from Firestore with improved error handling
def fetch_all_documents() -> None:
    try:
        collection_ref = db.collection('users')
        docs = collection_ref.stream()
        
        for doc in docs:
            user_id = doc.id
            user_data = doc.to_dict()
            
            # Validate user data
            if all(key in user_data for key in ['name', 'interests', 'userId']):
                user_interests_data[user_id] = {
                    'id': user_id,
                    'name': user_data['name'],
                    'interests': user_data['interests'],
                    'userId': user_data['userId']
                }
            else:
                logger.warning(f"Incomplete user data for user {user_id}")
        
        logger.info(f"Fetched {len(user_interests_data)} valid user documents")
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        user_interests_data.clear()

# Load the model with error handling
try:
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logger.info("Sentence Transformer model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

# Generate average embedding with memory-efficient approach
def generate_average_embedding(sentences: List[str]) -> np.ndarray:
    try:
        # Use model.encode with a smaller batch size and half-precision to reduce memory consumption
        embeddings = model.encode(sentences, batch_size=4, precision='float16')
        return np.mean(embeddings, axis=0, dtype=np.float16)
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float16)

# Initialize user embeddings with memory-aware approach
def initialize_user_embeddings() -> None:
    user_embeddings.clear()  # Clear previous embeddings
    
    for user_id, user_data in user_interests_data.items():
        try:
            interests = user_data['interests']
            user_embedding = generate_average_embedding(interests)
            user_embeddings[user_id] = {
                'embedding': user_embedding,
                'interests': interests
            }
        except Exception as e:
            logger.warning(f"Failed to generate embedding for user {user_id}: {e}")

# Find k-nearest neighbors with improved performance
def find_k_nearest_neighbors(user_id: str, k: int = 5) -> List[Tuple[str, List[str]]]:
    try:
        user_embedding = user_embeddings[user_id]['embedding']
        
        # Use generator expression and sort with key function
        similarities = sorted(
            ((other_user_id, np.dot(user_embedding, data['embedding']) / 
              (np.linalg.norm(user_embedding) * np.linalg.norm(data['embedding'])))
             for other_user_id, data in user_embeddings.items() 
             if other_user_id != user_id),
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(sim[0], user_embeddings[sim[0]]['interests']) for sim in similarities[:k]]
    except Exception as e:
        logger.error(f"Nearest neighbors calculation error: {e}")
        return []

# Add new user with logging
def add_new_user(user_id: str, interests_str: str) -> None:
    try:
        interests = [interest.strip() for interest in interests_str.split(',')]
        user_interests_data[user_id] = {
            'id': user_id,
            'name': f'User {user_id}',
            'interests': interests,
            'userId': user_id
        }
        
        user_embedding = generate_average_embedding(interests)
        user_embeddings[user_id] = {
            'embedding': user_embedding,
            'interests': interests
        }
        
        logger.info(f"Added new user {user_id} with {len(interests)} interests")
    except Exception as e:
        logger.error(f"Error adding new user {user_id}: {e}")

# Routes with improved error handling and logging
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_interests = request.form['user_interests']
            interests_list = [interest.strip() for interest in user_interests.split(',')]
            user_embedding = generate_average_embedding(interests_list)

            similarities = [
                (user_id, np.dot(user_embedding, data['embedding']) / 
                 (np.linalg.norm(user_embedding) * np.linalg.norm(data['embedding'])))
                for user_id, data in user_embeddings.items()
            ]
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_users = [(sim[0], user_embeddings[sim[0]]['interests']) for sim in similarities[:5]]

            new_user_id = f"user{len(user_interests_data) + 1}"
            add_new_user(new_user_id, user_interests)

            similar_users = [(user_id, interests) for user_id, interests in similar_users if user_id != new_user_id]

            return render_template('index.html', similar_users=similar_users)
        except Exception as e:
            logger.error(f"Error in index route: {e}")
            return render_template('index.html', error=str(e))

    return render_template('index.html')

@app.route('/api/similar_users', methods=['POST'])
def get_similar_users():
    try:
        fetch_all_documents()
        initialize_user_embeddings()
        
        data = request.get_json()
        user_interests = data.get('user_interests', '')
        interests_list = [interest.strip() for interest in user_interests.split(',')]
        user_embedding = generate_average_embedding(interests_list)

        similarities = [
            (user_id, np.dot(user_embedding, data['embedding']) / 
             (np.linalg.norm(user_embedding) * np.linalg.norm(data['embedding'])))
            for user_id, data in user_embeddings.items()
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = [{
            'id': user_interests_data[sim[0]]['id'],
            'name': user_interests_data[sim[0]]['name'],
            'interests': user_interests_data[sim[0]]['interests'],
            'userId': user_interests_data[sim[0]]['userId']
        } for sim in similarities[:10]]
        
        logger.info(f"Retrieved {len(similar_users)} similar users")
        return jsonify({'similar_users': similar_users})
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    try:
        logger.info("Starting application...")
        fetch_all_documents()
        initialize_user_embeddings()
        logger.info("Application initialized successfully")
        app.run(debug=True)
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")