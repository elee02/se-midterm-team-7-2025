import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import logging
from parser import parse_pdf, parse_text
from rag import RAG
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_cpp import Llama  # new import for llama.cpp

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device for inference: {device}")

# App setup
app = FastAPI(title="NotebookLM-like Tool")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Load smaller Qwen model for Hugging Face CPU - Use a non-FP8 variant
logger.info("Loading meta-llama/Llama-3.2-1B-Instruct model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
# Ensure model is explicitly on the chosen device
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True,
    # Optionally add torch_dtype=torch.float32 if memory allows, or let it default
).to(device)
logger.info(f"Model loaded onto device: {model.device}")

def generate_response(prompt: str) -> str:
    logger.debug(f"Generating response for prompt of length {len(prompt)}")
    # Tokenize and explicitly move inputs to CPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Log device of input tensors
    logger.debug(f"Input tensor device: {inputs['input_ids'].device}")
    logger.debug(f"Model device is {model.device}")
    logger.debug(f"Input shape: {inputs['input_ids'].shape}")
    logger.debug("Calling model.generate ...")
    try:
        # Ensure generation happens on CPU
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(**inputs, max_new_tokens=800, do_sample=True)  # Increased from 300 to 800 to generate longer responses
    except Exception as gen_err:
        logger.error(f"Generation error: {gen_err}")
        raise

    # Decode on CPU (tokenizer.decode works on CPU tensors)
    response = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    logger.debug("Model generation complete.")
    return response

# In-memory storage and RAG engine
rag = RAG()
documents = {}

class QueryRequest(BaseModel):
    question: str
    file_id: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Starting upload for: {file.filename}")
        logger.info(f"Received file: {file.filename}")
        if file.content_type not in ["application/pdf", "text/plain"]:
            raise HTTPException(status_code=400, detail="Only PDF or TXT allowed")

        file_id = str(len(documents) + 1)
        safe_name = file.filename.replace("/", "_").replace("\\", "_")
        file_path = f"uploads/{file_id}_{safe_name}"

        os.makedirs("uploads", exist_ok=True)
        file_bytes = await file.read()
        logger.debug(f"File size: {len(file_bytes)} bytes")

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        text = parse_pdf(file_bytes) if file.filename.endswith(".pdf") else parse_text(file_bytes)
        if not text.strip():
            text = "No extractable text found in this file."

        documents[file_id] = {"file_path": file_path, "file_name": file.filename}
        await rag.embed_document(file_id, text)

        logger.info(f"Completed upload for file_id={file_id}")
        return {"file_id": file_id, "file_name": file.filename}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process file")

@app.post("/generate")
async def query_file(request: QueryRequest):
    logger.info(f"Received query for file_id={request.file_id}")
    if request.file_id not in documents:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        context = await rag.query_document(request.question, request.file_id)
    except Exception as e:
        logger.error(f"Retrieval failed for file_id={request.file_id}: {e}")
        context = []

    if not context:
        logger.debug("No specific context, using general prompt")
        prompt = f"""The document doesn't contain specific information about this question, but please try to provide a helpful answer anyway:

Question: {request.question}

Please note that your answer is not based on the specific document content.
Do not repeat this prompt or the document details in your answer."""
    else:
        context_text = "\n\n".join(context)
        prompt = f"""You are an assistant that helps users understand document content.
        
Below are relevant passages from the document '{documents[request.file_id]["file_name"]}':

{context_text}

Based on the above passages only, please answer this question:
{request.question}

Do not include or repeat the document passages in your answer. If the answer isn't found in the passages, say "I don't see information about this in the provided document sections." Then try to give a helpful general response."""

    try:
        answer = generate_response(prompt)
        logger.info(f"Model generated answer: {answer}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Error generating answer")

    return {"answer": answer, "context": context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)