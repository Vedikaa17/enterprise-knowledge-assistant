import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# Step 2: Load documents and create chunks
documents_folder = "documents"
all_texts = []

for filename in os.listdir(documents_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(documents_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        all_texts.append({
            "filename": filename,
            "content": text
        })


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []

for doc in all_texts:
    chunks = text_splitter.split_text(doc["content"])

    for chunk in chunks:
        all_chunks.append({
            "source": doc["filename"],
            "text": chunk
        })

print(f"Total chunks to embed: {len(all_chunks)}")


# Step 3: Create Chroma client and collection
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="enterprise_docs"
)


# Step 4: Generate embeddings and store in Chroma
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
        metadatas=[
            {
                "source": chunk["source"]
            }
        ]
    )

    print(
        f"Stored chunk {i + 1}/{len(all_chunks)} "
        f"from {chunk['source']}"
    )


print("\nAll chunks embedded and stored successfully!")