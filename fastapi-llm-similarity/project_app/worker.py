from sentence_transformers import SentenceTransformer
import chromadb
from project_app.config import MODEL_NAME, CHROMA_PATH

model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="projects")

def find_similar_projects(description: str, threshold: float):
    """Background task for similarity search."""
    embedding = model.encode(description).tolist()

    results = collection.query(query_embeddings=[embedding], n_results=10)
    project_ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []
    for pid, dist in zip(project_ids, distances):
        similarity = 1 - dist  # Chroma returns cosine distance
        if similarity >= threshold:
            matches.append({"project_id": pid, "similarity": round(similarity, 3)})

    return matches
    
