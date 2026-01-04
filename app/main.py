from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router

app = FastAPI(
    title="Chatbot Juridique Tunisien",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

"""LANCER LE BACKEND
uvicorn app.main:app --reload


➡️ Swagger auto :
👉 http://127.0.0.1:8000/docs"""

#Une startup peut-elle choisir librement le commissaire aux apports pour un apport en nature ?