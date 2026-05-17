from fastapi import FastAPI
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

app = FastAPI(title="Kişisel Yapay Zeka API")

# Yeni ve güncel kütüphane ile modelimize bağlanıyoruz
llm = OllamaLLM(model="llama3")

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = llm.invoke(request.prompt)
    return {"response": response}