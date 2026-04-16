from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from index_dataset import load_existing_index
from Rag_engine import ask_gem_guide, get_artifact_metadata
import time
import psutil
from utils import get_gpu_stats

resources = {}

@asynccontextmanager
async def startup(app: FastAPI):
    index, chroma_collection = load_existing_index()
    resources["index"] = index
    resources["chroma"] = chroma_collection
    yield

app = FastAPI(title="GEM Tour RAG API", lifespan=startup)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question:str
    artifact_id:str | None = None

@app.post("/ask")
def ask_guide(request: QueryRequest):
    try:
        result = ask_gem_guide(
            resources["index"],
            resources["chroma"],
            "i82blikeu/gemma-3n-E4B-it-GGUF:Q3_K_M",
            request.question,
            request.artifact_id,
            verbose=False
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifact/{artifact_id}")
def get_artifact(artifact_id: str):
    metadata = get_artifact_metadata(resources["chroma"], artifact_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return metadata

###################################################################

MODELS = {
    "gemma3n-e4b": "i82blikeu/gemma-3n-E4B-it-GGUF:Q3_K_M",
    "gemma4-e2b-q4": "batiai/gemma4-e2b:q4",
    "qwen3-4b": "qwen3:4b",
    # "qwen3.5-4b": "qwen3.5:4b",
    "llama3.2-3b": "llama3.2:3b",
}

# test_questions = [
#     # Camera mode - English
#     {
#         "question": "Tell me about this artifact, what is it made of and where was it found?",
#         "artifact_id": "Mask of Tutankhamun"
#     },
#     # Camera mode - Arabic
#     {
#         "question": "من هو صاحب هذا التمثال وما قصته؟",
#         "artifact_id": "Seated Statue of King Amenhotep III"
#     },
#     # Camera mode - English
#     {
#         "question": "Who is this person and what was his role?",
#         "artifact_id": "Vizier Khaemwaset"
#     },
#     # Camera mode - Arabic
#     {
#         "question": "أخبرني عن هذه الملكة وما هي إنجازاتها؟",
#         "artifact_id": "Kneeling Statue of Queen Hatshepsut"
#     },
#     # Text mode - English
#     {
#         "question": "Which pharaohs from Dynasty 18 are represented in the museum?",
#         "artifact_id": None
#     },
#     # Text mode - English
#     {
#         "question": "Tell me about the role of high priests in ancient Egypt",
#         "artifact_id": None
#     },
# ]

# class BenchmarkRequest(BaseModel):
#     question: str
#     artifact_id: str | None = None

test_questions = [
    # Camera mode - English
    {
        "question": "Tell me about this artifact, what is it made of and where was it found?",
        "artifact_id": "Mask of Tutankhamun"
    },
    # Camera mode - Arabic
    {
        "question": "من هو صاحب هذا التمثال وما قصته؟",
        "artifact_id": "Seated Statue of King Amenhotep III"
    },
    # Camera mode - English
    {
        "question": "Who is this person and what was his role?",
        "artifact_id": "Vizier Khaemwaset"
    },
    # Camera mode - Arabic
    {
        "question": "أخبرني عن هذه الملكة وما هي إنجازاتها؟",
        "artifact_id": "Kneeling Statue of Queen Hatshepsut"
    },
    # Text mode - English
    {
        "question": "Which pharaohs from Dynasty 18 are represented in the museum?",
        "artifact_id": None
    },
    # Text mode - English
    {
        "question": "Tell me about the role of high priests in ancient Egypt",
        "artifact_id": None
    },
    # Camera mode - English
    {
        "question": "What is the historical significance of this statue?",
        "artifact_id": "Colossal Statue of King Senwosret III"
    },
    # Camera mode - Arabic
    {
        "question": "ما هي المواد المصنوع منها هذا التمثال ومن أين جاء؟",
        "artifact_id": "Ramesses II Seated Statue"
    },
    # Text mode - Arabic
    {
        "question": "ما هو دور الكتبة في مصر القديمة؟",
        "artifact_id": None
    },
    # Text mode - English
    {
        "question": "What artifacts from Karnak are displayed in the museum?",
        "artifact_id": None
    },
]

@app.get("/benchmark")
def benchmark():
    results = []
    for test_question in test_questions:
        for model_label, model_name in MODELS.items():
            process = psutil.Process()
            gpu_before = get_gpu_stats()
            mem_before = process.memory_info().rss / 1024 / 1024

            start = time.time()
            result = ask_gem_guide(
                resources["index"],
                resources["chroma"],
                model_name,
                test_question['question'],
                test_question['artifact_id'],
                verbose=False
            )
            elapsed = time.time() - start

            gpu_after = get_gpu_stats()
            mem_after = process.memory_info().rss / 1024 / 1024

            results.append({
                "question": test_question['question'],
                "model": model_label,
                "time_seconds": round(elapsed, 2),
                "ram_delta_mb": round(mem_after - mem_before, 2),
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "vram_delta_mb": (gpu_after["vram_used_mb"] - gpu_before["vram_used_mb"]) if gpu_before and gpu_after else None,
                "response_length": len(result["raw_response"]),
                "response": result["raw_response"],
            })

    return {"results": results}