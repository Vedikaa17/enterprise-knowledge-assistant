import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

# Step 1: API key load karo, Gemini client banao
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 2: Chroma se hamari existing collection connect karo
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def retrieve_relevant_chunks(query, top_k=3):
    # Step 3: User ke question ka embedding banao (RETRIEVAL_QUERY type)
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    query_embedding = result.embeddings[0].values

    # Step 4: Chroma mein sabse similar chunks dhoondo
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

# Step 5: Test karo
if __name__ == "__main__":
    user_question = "How many leave days do employees get?"
    results = retrieve_relevant_chunks(user_question)

    print(f"Question: {user_question}\n")
    print("Top matching chunks:\n")

    for i in range(len(results["documents"][0])):
        chunk_text = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]

        print(f"Result {i+1} (from {source}, distance: {distance:.4f})")
        print(chunk_text)
        print("-" * 50)