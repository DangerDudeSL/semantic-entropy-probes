
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from engine import engine
import uvicorn
import contextlib

# Lifecycle manager for startup/shutdown
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        engine.load_probes()
        success = engine.load_model()
        if not success:
            print("WARNING: Model failed to load on startup.")
    except Exception as e:
        print(f"Startup error: {e}")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS (Allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class SentenceDetail(BaseModel):
    text: str
    entropy: float
    accuracy_prob: float

class QueryResponse(BaseModel):
    answer: str
    entropy: float
    accuracy_prob: float
    sentence_details: list[SentenceDetail] = []
    error: str = None

@app.get("/status")
def get_status():
    return {
        "model_loaded": engine.model is not None,
        "probes_loaded": engine.probes is not None,
        "model_name": engine.model_name,
        "probe_name": engine.selected_probe['name'] if engine.selected_probe else "None"
    }

class SetModelRequest(BaseModel):
    model_name: str

@app.post("/set_model")
def set_model(request: SetModelRequest):
    if engine.model_name == request.model_name and engine.model is not None:
         return {"status": "Already loaded"}
         
    # Unload existing
    engine.unload_model()
    
    # Load new
    success = engine.load_model(request.model_name)
    if not success:
         raise HTTPException(status_code=500, detail="Failed to load model")
         
    return {"status": "Model changed", "model_name": request.model_name}

@app.post("/infer", response_model=QueryResponse)
async def infer(request: QueryRequest):
    if not engine.model:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    # Run inference in threadpool to avoid blocking async loop (since model.generate is blocking)
    # Using asyncio.to_thread for PyTorch operations
    result = await asyncio.to_thread(engine.generate_response, request.question)
    
    if "error" in result:
         # Still return 200 OK so frontend can display the error nicely
         return QueryResponse(answer="", entropy=0.0, accuracy_prob=0.0, error=result["error"])
         
    return QueryResponse(
        answer=result["answer"],
        entropy=result["entropy"],
        accuracy_prob=result["accuracy_prob"],
        sentence_details=result.get("sentence_details", [])
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
