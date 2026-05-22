import os
import random
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from dotenv import load_dotenv

# RAG Stack
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# Constants
PDF_PATH = os.path.join(settings.BASE_DIR, "data", "24pagepdf.pdf")
INDEX_PATH = os.path.join(settings.BASE_DIR, "faiss_index")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# List of models to try in order (70B is best, 8B is fastest/less likely to hit limits)
MODEL_LIST = [
    "llama-3.3-70b-versatile", 
    "llama-3.1-70b-versatile", 
    "llama3-70b-8192", 
    "llama-3.3-70b-specdec",
    "llama3-8b-8192",         # Fallback to smaller model if 70B is blocked
    "mixtral-8x7b-32768"      # Another alternative architecture
]

def get_vector_db():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    return vector_db

def chat_home(request):
    return render(request, 'chat/index.html')

def get_response(request):
    if request.method == "POST":
        query = request.POST.get('message')
        
        try:
            # 1. Retrieve Context
            db = get_vector_db()
            search_results = db.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in search_results])
            
            prompt = f"""
            You are Orewa Bot, the Academy Advisor. ✨
            Answer ONLY using the PDF context below. If not found, say we don't offer it.
            
            CONTEXT:
            {context}
            
            USER QUESTION:
            {query}
            """

            # 2. Generate with Fallback Logic
            ai_message = None
            last_error = ""

            for model_id in MODEL_LIST:
                try:
                    llm = ChatGroq(
                        model=model_id, 
                        temperature=0.3,
                        api_key=os.getenv("GROQ_API_KEY")
                    )
                    ai_message = llm.invoke(prompt)
                    # If we get here, the model worked! Break the loop.
                    break 
                except Exception as e:
                    print(f"Model {model_id} failed: {str(e)}")
                    last_error = str(e)
                    continue # Try the next model in the list

            if ai_message:
                return JsonResponse({'reply': ai_message.content})
            else:
                return JsonResponse({'reply': f"I'm currently overwhelmed! All models failed. Error: {last_error}"}, status=429)

        except Exception as e:
            return JsonResponse({'reply': f"General Error: {str(e)}"}, status=500)