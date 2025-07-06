import os
import json
import math
import random
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
# from sklearn.metrics.pairwise import cosine_similarity
# import cv2
# import face_recognition
from starlette.websockets import WebSocket
from starlette.middleware.cors import CORSMiddleware
import socketio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


app = Flask(__name__)
# app = FastAPI()
CORS(app)  # Enable CORS for all routes
# sio = SocketIO(app, cors_allowed_origins="*")

# sio = socketio.AsyncServer()
# # app = FastAPI()

# app.add_middleware(CORSMiddleware, allow_origins=["*"])


# Configuration
DATA_DIR = "data"
CLUSTER_COUNT = 500
MIN_CLUSTER_SIZE = 5
MAX_CLUSTER_SIZE = 100
MIN_SIMILARITY = 0.6  # Minimum similarity between adjacent images
SIMILARITY_VARIANCE = 0.2  # How much similarity can vary
FACENET_MODEL = "large"  # "small" for faster but less accurate, "large" for better accuracy
IMAGE_BASE_URL = "https://randomuser.me/api/portraits/"

# Generate face embedding using FaceNet
def get_face_embedding(image_path):
    # In a real system, you'd load the image from storage
    # For demo, generate a random embedding
    return np.random.rand(128).tolist()

# # Calculate similarity between two face embeddings
# def calculate_similarity(embedding1, embedding2):
#     # Convert to numpy arrays
#     emb1 = np.array(embedding1).reshape(1, -1)
#     emb2 = np.array(embedding2).reshape(1, -1)
    
#     # Calculate cosine similarity
#     similarity = cosine_similarity(emb1, emb2)[0][0]
#     return max(0, min(1, similarity))  # Clamp between 0 and 1

# Generate mock cluster data
def generate_cluster_data():
    clusters = []
    for i in range(CLUSTER_COUNT):
        cluster_size = random.randint(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
        avg_similarity = random.uniform(0.75, 0.95)
        
        clusters.append({
            "id": f"C{str(i).zfill(4)}",
            "size": cluster_size,
            "avg_similarity": round(avg_similarity, 2),
            "position": i  # The starting row position
        })
    
    # Sort by size descending for bookshelf visualization
    clusters.sort(key=lambda x: x["size"], reverse=True)
    
    # Assign actual row positions based on sorted order
    current_y = 0
    for cluster in clusters:
        cluster["position"] = current_y
        current_y += 140  # Each row height is 140px
    
    return clusters

# Generate image data for a cluster with similarity calculations
def generate_cluster_images(cluster_id, cluster_size):
    images = []
    # Generate base embeddings for the cluster
    base_embedding = get_face_embedding(None)
    embeddings = [base_embedding]
    
    for i in range(1, cluster_size):
        # Create a slight variation of the previous embedding
        new_embedding = embeddings[-1] + (np.random.rand(128) - 0.5) * 0.1
        embeddings.append(new_embedding.tolist())
    
    for i in range(cluster_size):
        # Generate similarity to previous image
        if i == 0:
            left_similarity = None
        else:
            similarity = 1 #calculate_similarity(embeddings[i-1], embeddings[i])
            left_similarity = similarity
            
            # Add some variance for realism
            left_similarity = max(MIN_SIMILARITY, min(0.99, 
                                 left_similarity + random.uniform(-SIMILARITY_VARIANCE, SIMILARITY_VARIANCE)))
        
        # For the last image, no right similarity
        right_similarity = None if i == cluster_size - 1 else 1 #calculate_similarity(embeddings[i], embeddings[i+1])
        
        # Generate mock image path
        gender = "men" if random.random() > 0.5 else "women"
        img_id = random.randint(1, 99)
        img_url = f"{IMAGE_BASE_URL}/{gender}/{img_id}.jpg"
        
        images.append({
            "id": f"F{str(i).zfill(4)}",
            "cluster_id": cluster_id,
            "path": img_url,
            "left_similarity": round(left_similarity, 3) if left_similarity else None,
            "right_similarity": round(right_similarity, 3) if right_similarity else None,
            "position": i  # Position in cluster sequence
        })
    
    return images

# Create mock data files
def create_mock_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Generate and save cluster data
    clusters = generate_cluster_data()
    with open(os.path.join(DATA_DIR, "clusters.json"), "w") as f:
        json.dump(clusters, f)
    
    # Generate and save image data for each cluster
    for cluster in clusters:
        images = generate_cluster_images(cluster["id"], cluster["size"])
        with open(os.path.join(DATA_DIR, f"cluster_{cluster['id']}.json"), "w") as f:
            json.dump(images, f)

# API Endpoints
@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    # Load cluster data
    with open(os.path.join(DATA_DIR, "clusters.json"), "r") as f:
        clusters = json.load(f)
    
    # Apply filters if provided
    min_size = request.args.get('min_size', default=1, type=int)
    max_size = request.args.get('max_size', default=1000, type=int)
    search = request.args.get('search', default="", type=str)
    filter = request.args.get('filter', default=None, type=str)
    similarity_threshold = request.args.get('similarity_threshold', default=None, type=float)
    
    if filter:
        if filter == "small":
            min_size = 1
            max_size = 5
        elif filter == "medium":
            min_size = 6
            max_size = 20
        elif filter == "large":
            min_size = 21
            max_size = 40
        elif filter == "all":
            min_size = 1
            max_size = 1000

    if similarity_threshold is not None:
        clusters = [c for c in clusters if c["avg_similarity"] >= similarity_threshold]

    filtered_clusters = [
        c for c in clusters 
        if min_size <= c["size"] <= max_size and 
        search.lower() in c["id"].lower()
    ]
    
    # Pagination
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=50, type=int)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    paginated_clusters = filtered_clusters[start_idx:end_idx]
    
    return jsonify({
        "clusters": paginated_clusters,
        "total_clusters": len(filtered_clusters),
        "page": page,
        "per_page": per_page,
        "total_pages": math.ceil(len(filtered_clusters) / per_page)
    })

@app.route('/api/cluster/<cluster_id>', methods=['GET'])
def get_cluster(cluster_id):
    try:
        with open(os.path.join(DATA_DIR, f"cluster_{cluster_id}.json"), "r") as f:
            images = json.load(f)
        
        # Apply pagination for images within a cluster
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('per_page', default=20, type=int)
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paginated_images = images[start_idx:end_idx]
        
        return jsonify({
            "cluster_id": cluster_id,
            "images": paginated_images,
            "total_images": len(images),
            "page": page,
            "per_page": per_page,
            "total_pages": math.ceil(len(images) / per_page)
        })
    except FileNotFoundError:
        return jsonify({"error": "Cluster not found"}), 404

@app.route('/api/merge_clusters', methods=['POST'])
def merge_clusters():
    data = request.json
    cluster_ids = data.get('cluster_ids', [])
    if not cluster_ids:
        return jsonify({"error": "No clusters provided"}), 400
    
    # In a real implementation, this would merge clusters in the database
    # For mock purposes, return a new cluster ID
    new_cluster_id = f"C{random.randint(1000, 9999)}"
    
    # # Send real-time update
    # sio.emit("cluster_update", {
    #     "type": "cluster_merged",
    #     "new_cluster_id": new_cluster_id,
    #     "merged_cluster_ids": cluster_ids
    # })
    
    return jsonify({
        "message": "Clusters merged successfully",
        "new_cluster_id": new_cluster_id
    })

@app.route('/api/split_cluster', methods=['POST'])
def split_cluster():
    data = request.json
    cluster_id = data.get('cluster_id')
    image_ids = data.get('image_ids', [])
    
    if not cluster_id or not image_ids:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # In a real implementation, this would split the cluster
    # For mock purposes, return a new cluster ID
    new_cluster_id = f"C{random.randint(1000, 9999)}"
    
    # Send real-time update
    # sio.emit("cluster_update", {
    #     "type": "cluster_split",
    #     "new_cluster_id": new_cluster_id,
    #     "source_cluster_id": cluster_id,
    #     "image_ids": image_ids
    # })
    
    return jsonify({
        "message": "Cluster split successfully",
        "new_cluster_id": new_cluster_id
    })

#serve the static html files
@app.route('/')
def index():
    return send_from_directory('.', 'frontend.html')

# # WebSocket initialization
# @sio.on('connect')
# def handle_connect():
#     print('Client connected')
#     sio.emit("status", {"message": "Connected to Face Cluster API"})

if __name__ == '__main__':
    # Create mock data if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, "clusters.json")):
        print("Generating mock data...")
        create_mock_data()
    
    print("Starting server on port 5000")
    # sio.async_mode = "asyncio"
    # sio.run(app, host='0.0.0.0', port=5000, debug=True)
    # sio.run(app, debug=True, host='0.0.0.0', port=5000)
    app.run( host='0.0.0.0', port=5000)