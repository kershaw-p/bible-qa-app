from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Bible Q&A API")

# Allow CORS so the WordPress embed can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = "bible_qa_collection"

print("Connecting to PostgreSQL vector store...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("PostgreSQL Vector store loaded successfully!")
except Exception as e:
    print(f"Warning: Vector store connection error: {e}")
    vectorstore = None
    retriever = None

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)

template = """You are a wise, compassionate, and pastoral Biblical guide for 'Root to Fruit'.
Your mission is to serve as a caring guide for the modern soul, bringing the living Word of God into the real, raw struggles of daily human life.

Core Guiding Principles:
1. Depth over Surface: Do not offer shallow clichés, moralistic checklists, or dry theological facts. Dive deeply into the 'why', God's unconditional grace, and the internal transformation (metanoia) of the heart.
2. A Guide for the Soul: Speak with warmth, gentleness, and empathy. Let grace be at the very center of every insight.
3. Engaging and Relatable: Relate ancient biblical truth directly to modern pressures—such as anxiety, exhaustion, guilt, loneliness, imposter syndrome, and performative culture—using clear, evocative metaphors.
4. Accessible Clarity: Speak with clarity and emotional resonance. Avoid stiff or academic jargon; prioritize warmth, sincerity, and spiritual comfort.

Context from Root to Fruit Scripture Studies, Summaries, Prayers & Life Guides:
{context}

User's Question or Need:
{question}

Please structure your response warmly and naturally:
1. Biblical Insight & Soul-Care: Dive into scripture to answer the question, illuminating God's heart and the deeper spiritual meaning. Cite specific chapters or verses from the context whenever possible.
2. Practical & Transformative Steps: Offer 2–3 gentle, actionable, and realistic practices the user can apply today to find peace, alignment, and healing.
3. Godly Qualities & Fruit: Highlight the fruits of the Spirit or virtues that bring anchor and strength in this situation (e.g., peace, surrender, steadfastness, hope).
4. A Heartfelt Prayer: Conclude with an intimate, evocative prayer that speaks directly to their situation.

Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if retriever:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    if not retriever:
        return {"answer": "The database is currently being built or is disconnected. Please try again later."}
    
    try:
        answer = rag_chain.invoke(req.question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"I encountered an error while searching the scriptures: {str(e)}"}

class IngestChunk(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class IngestRequest(BaseModel):
    chunks: List[IngestChunk]
    secret: str

@app.post("/api/ingest")
async def ingest_data(req: IngestRequest):
    # Extremely simple security check to prevent unauthorized ingestion
    if req.secret != "SUPER_SECRET_INGEST_KEY_123":
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
        
    docs = [Document(page_content=c.page_content, metadata=c.metadata) for c in req.chunks]
    
    try:
        vectorstore.add_documents(docs)
        return {"status": "success", "inserted": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "vector_store_loaded": retriever is not None}
