"""
RAG Document Q&A — FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.database import init_db

app = FastAPI(
    title="RAG Document Q&A",
    description="FAANG-grade RAG pipeline: hybrid retrieval + reranking + citations + RAGAS eval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    """Initialize DB on startup."""
    init_db()
    print("[App] Database initialized")


@app.get("/")
async def root():
    return {
        "name": "RAG Document Q&A API",
        "version": "1.0.0",
        "docs": "/docs"
    }
