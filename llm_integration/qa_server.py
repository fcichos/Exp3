"""
Q&A API Server for Experimental Physics 3 Course
This server provides an API for question-answering using embedded course content
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import asyncio
from datetime import datetime
import logging

# Web framework
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Vector database and LLM libraries
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "fastapi", "uvicorn", "langchain", "chromadb",
                          "sentence-transformers", "transformers", "torch", "accelerate"])
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Experimental Physics 3 Q&A API",
    description="API for answering questions about the Experimental Physics 3 course content",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    context_size: int = Field(default=5, description="Number of relevant documents to retrieve")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: int = Field(default=500, description="Maximum tokens in response")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict] = Field(..., description="Source documents used")
    timestamp: str = Field(..., description="Timestamp of the response")
    confidence: float = Field(..., description="Confidence score of the answer")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_loaded: bool
    timestamp: str

# Global variables for models
embeddings_model = None
vectorstore = None
qa_chain = None
llm_pipeline = None

def initialize_embeddings():
    """Initialize the embedding model"""
    global embeddings_model
    if embeddings_model is None:
        logger.info("Loading embedding model...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model loaded successfully")
    return embeddings_model

def initialize_vectorstore(persist_directory: str = "./chroma_db_exp3"):
    """Initialize the vector store"""
    global vectorstore
    if vectorstore is None:
        logger.info("Loading vector store...")
        if not os.path.exists(persist_directory):
            raise ValueError(f"Vector database not found at {persist_directory}. "
                           "Please run website_embedder.py first to create the database.")

        embeddings = initialize_embeddings()
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logger.info("Vector store loaded successfully")
    return vectorstore

def initialize_llm():
    """Initialize the language model for Q&A"""
    global llm_pipeline
    if llm_pipeline is None:
        logger.info("Loading language model...")

        # Using a smaller model for efficiency
        # You can replace with other models like "google/flan-t5-base" or "microsoft/phi-2"
        model_name = "google/flan-t5-small"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )

            # Create pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7
            )

            llm_pipeline = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Language model {model_name} loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load advanced model: {str(e)}")
            logger.info("Falling back to simple retrieval-based responses")
            llm_pipeline = None

    return llm_pipeline

def initialize_qa_chain():
    """Initialize the QA chain"""
    global qa_chain
    if qa_chain is None:
        logger.info("Initializing QA chain...")

        vectorstore = initialize_vectorstore()
        llm = initialize_llm()

        if llm:
            # Create prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always include which topic or chapter the information comes from if available.

Context:
{context}

Question: {question}

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        else:
            # Fallback to simple retrieval
            qa_chain = None

        logger.info("QA chain initialized")

    return qa_chain

def simple_answer_generation(question: str, documents: List[Dict]) -> str:
    """
    Generate a simple answer based on retrieved documents (fallback method)
    """
    if not documents:
        return "I couldn't find relevant information to answer your question."

    # Combine the top documents
    context = "\n\n".join([doc['content'] for doc in documents[:3]])

    # Simple template-based response
    answer = f"""Based on the course materials, here's what I found relevant to your question:

{context[:1000]}

This information comes from the following sources:
"""

    for doc in documents[:3]:
        answer += f"\n- {doc['title']}"

    return answer

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        initialize_embeddings()
        initialize_vectorstore()
        # Optional: initialize LLM (can be done on first request to save memory)
        # initialize_llm()
        # initialize_qa_chain()
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=llm_pipeline is not None,
        vectorstore_loaded=vectorstore is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a question based on the course content
    """
    try:
        # Initialize vectorstore if needed
        vs = initialize_vectorstore()

        # Perform similarity search
        results = vs.similarity_search_with_score(
            request.question,
            k=request.context_size
        )

        # Format documents
        documents = []
        for doc, score in results:
            documents.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'title': doc.metadata.get('title', 'Unknown'),
                'score': float(score)
            })

        # Generate answer
        try:
            # Try to use QA chain if available
            chain = initialize_qa_chain()
            if chain:
                response = chain({"query": request.question})
                answer = response["result"]
                confidence = 0.8  # High confidence with LLM
            else:
                # Fallback to simple answer generation
                answer = simple_answer_generation(request.question, documents)
                confidence = 0.6  # Medium confidence without LLM

        except Exception as e:
            logger.warning(f"Failed to generate answer with LLM: {str(e)}")
            answer = simple_answer_generation(request.question, documents)
            confidence = 0.5  # Lower confidence with fallback

        # Prepare response
        return AnswerResponse(
            answer=answer,
            sources=documents[:3],  # Return top 3 sources
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
async def list_sources():
    """List all available sources in the vector database"""
    try:
        vs = initialize_vectorstore()

        # Get all documents (limited sample)
        results = vs.similarity_search("", k=100)

        # Extract unique sources
        sources = set()
        for doc in results:
            source = doc.metadata.get('source', 'Unknown')
            title = doc.metadata.get('title', 'Unknown')
            sources.add(f"{title} ({source})")

        return {
            "sources": sorted(list(sources)),
            "count": len(sources),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(question: str, answer: str, helpful: bool, comments: Optional[str] = None):
    """
    Submit feedback on an answer (for future improvements)
    """
    feedback_dir = Path("./feedback")
    feedback_dir.mkdir(exist_ok=True)

    feedback_data = {
        "question": question,
        "answer": answer,
        "helpful": helpful,
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }

    # Save feedback to file
    feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)

    return {"status": "success", "message": "Thank you for your feedback!"}

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=True)

if __name__ == "__main__":
    run_server()
