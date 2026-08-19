import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Step 1: Setup - API key, Gemini client, Chroma connection
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

# Step 2: FastAPI app banao
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Step 3: Request body ka structure define karo
class QuestionRequest(BaseModel):
    question: str


def retrieve_relevant_chunks(query, top_k=3):
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    query_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def build_prompt(query, retrieved_chunks):
    context_parts = []
    for i in range(len(retrieved_chunks["documents"][0])):
        chunk_text = retrieved_chunks["documents"][0][i]
        source = retrieved_chunks["metadatas"][0][i]["source"]
        context_parts.append(f"[Source: {source}]\n{chunk_text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based only on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information in the context above.
- If the answer is not in the context, say "I don't have enough information to answer this."
- After your answer, mention which source file(s) you used, like: (Source: filename.txt)

Answer:"""

    return prompt


# Step 4: Root endpoint (health check)
@app.get("/")
def read_root():
    return {"message": "Enterprise Knowledge Assistant is running!"}


# Step 5: Actual /ask endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    query = request.question

    retrieved_chunks = retrieve_relevant_chunks(query)
    prompt = build_prompt(query, retrieved_chunks)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return {
        "question": query,
        "answer": response.text
    }