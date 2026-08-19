import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Sab documents ka text ek list mein load karo
documents_folder = "documents"
all_texts = []

for filename in os.listdir(documents_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(documents_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            all_texts.append({"filename": filename, "content": text})

print(f"Total documents loaded: {len(all_texts)}")

# Step 2: Splitter banao - chunk size aur overlap decide karo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # har chunk ~500 characters ka
    chunk_overlap=50     # chunks ke beech thoda overlap, context na tute isliye
)

# Step 3: Har document ko chunks mein todo
all_chunks = []
for doc in all_texts:
    chunks = text_splitter.split_text(doc["content"])
    for chunk in chunks:
        all_chunks.append({
            "source": doc["filename"],
            "text": chunk
        })

print(f"Total chunks created: {len(all_chunks)}")
print("\n--- Sample chunk ---")
print(all_chunks[0])