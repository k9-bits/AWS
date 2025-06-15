import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, status, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI  # Synchronous client

# --- 1. Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Security: Load the Server's Secret API Key ---
API_KEY = os.getenv("MY_API_KEY")

# --- 3. FastAPI Application Setup ---
app = FastAPI(
    title="Gemma 3 API",
    description="Synchronous FastAPI frontend for vLLM backend with streaming.",
    version="1.1.0"
)

# --- 4. OpenAI Client for vLLM Backend ---
client = OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="not-needed"  # vLLM doesn't require this
)

# --- 5. Security Dependency ---
def get_api_key(x_api_key: str = Header(..., description="The client's secret API key.")):
    if not API_KEY:
        logger.error("CRITICAL: MY_API_KEY not set.")
        raise HTTPException(status_code=500, detail="API Key not configured.")
    if x_api_key != API_KEY:
        logger.warning("Invalid API key attempt.")
        raise HTTPException(status_code=403, detail="Invalid API Key.")
    return x_api_key

# --- 6. Request Body Model ---
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

# --- 7. Streaming Generator ---
def stream_vllm_response(request: PromptRequest):
    messages = [{"role": "user", "content": request.prompt}]
    try:
        response = client.chat.completions.create(
            model="ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g",
            messages=messages,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True
        )

        for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    except Exception as e:
        yield f"\n[Error streaming response: {str(e)}]\n"

# --- 8. Streaming Endpoint ---
@app.post("/generate-stream", dependencies=[Depends(get_api_key)])
def generate_text_stream(request: PromptRequest):
    """
    Streams generated text token-by-token from the vLLM backend.
    """
    return StreamingResponse(
        stream_vllm_response(request),
        media_type="text/plain"
    )

# --- 9. Non-Streaming Endpoint ---
@app.post("/generate", dependencies=[Depends(get_api_key)])
def generate_text(request: PromptRequest):
    """
    Forwards a prompt to the vLLM backend and returns the full response.
    """
    try:
        messages = [{"role": "user", "content": request.prompt}]
        logger.info(f"Sending request to vLLM: '{request.prompt[:70]}...'")

        response = client.chat.completions.create(
            model="ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g",
            messages=messages,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        result = response.choices[0].message.content.strip()
        logger.info("Received response successfully.")
        return {"generated_text": result}

    except Exception as e:
        logger.error(f"vLLM error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Error communicating with the model server."
        )

# --- 10. Startup Event ---
@app.on_event("startup")
def startup_event():
    logger.info("Starting Gemma API server...")
    if not API_KEY:
        logger.warning("!!! MY_API_KEY is not set. Requests will be rejected.")
    else:
        logger.info("API Key loaded. Server ready.")

# --- 11. Run ---
if __name__ == "__main__":
    uvicorn.run("gemma:app", host="0.0.0.0", port=8000, reload=True)
