import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def embed_documents_if_needed():
    """Agar Chroma collection khali hai, to documents ko chunk + embed karke store karo."""
    existing_count = collection.count()

    if existing_count > 0:
        print(f"Collection already has {existing_count} items. Skipping embedding.")
        return

    print("Collection is empty. Embedding documents now...")

    documents_folder = "documents"
    all_texts = []

    for filename in os.listdir(documents_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(documents_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            all_texts.append({"filename": filename, "content": text})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_chunks = []
    for doc in all_texts:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append({"source": doc["filename"], "text": chunk})

    for i, chunk in enumerate(all_chunks):
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk["text"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        embedding = result.embeddings[0].values

        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )

    print(f"Embedded and stored {len(all_chunks)} chunks.")


# Step 4: Startup event - server start hote hi ye chalega
@app.on_event("startup")
def on_startup():
    embed_documents_if_needed()


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


# Step 5: Root endpoint (health check)
@app.get("/")
def read_root():
    return {"message": "Enterprise Knowledge Assistant is running!"}


# Step 6: Actual /ask endpoint
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