from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse
from app.rag.hybrid_rag import hybrid_rag_answer

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = hybrid_rag_answer(req.question)
    return {"answer": answer}
