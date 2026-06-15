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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

template = """You are a wise, compassionate, and highly knowledgeable Biblical assistant for 'Root to Fruit'.
Your goal is to help users find answers to their questions or problems using explicitly what is said in the Bible.
Use the provided Biblical context (which includes verses, explanations, and life lessons) to answer the user's question.

If the answer is not contained in the context, you can draw upon your general knowledge of the Bible, but ALWAYS prioritize the provided context and cite specific chapters or verses mentioned.
Speak with an encouraging and uplifting tone.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

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
