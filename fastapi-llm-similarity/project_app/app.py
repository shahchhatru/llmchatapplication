from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import redis
from rq import Queue
from project_app.worker import find_similar_projects
from project_app.config import MODEL_NAME, CHROMA_PATH, REDIS_URL

# Initialize FastAPI
app = FastAPI(title="Project Similarity API (with Poetry)")

# Initialize model + chroma
model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="projects")

# Redis Queue
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue("similarity_tasks", connection=redis_conn)


class ProjectInfo(BaseModel):
    project_id: str
    description: str


class SimilarityRequest(BaseModel):
    description: str
    threshold: float = 0.9


@app.post("/add_project")
def add_project(data: ProjectInfo):
    """Store project and its embedding in Chroma."""
    embedding = model.encode(data.description).tolist()
    collection.add(ids=[data.project_id], documents=[data.description], embeddings=[embedding])
    return {"status": "success", "project_id": data.project_id}


@app.post("/find_similar")
def find_similar(data: SimilarityRequest):
    """Queue job for similarity check."""
    job = task_queue.enqueue(find_similar_projects, data.description, data.threshold)
    return {"status": "queued", "job_id": job.id}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    """Check queued job result."""
    job = task_queue.fetch_job(job_id)
    if not job:
        return {"error": "Invalid job ID"}
    if job.is_finished:
        return {"status": "completed", "result": job.result}
    elif job.is_failed:
        return {"status": "failed"}
    else:
        return {"status": "processing"}
