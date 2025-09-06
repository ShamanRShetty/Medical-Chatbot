import os
import hashlib
import json
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def load_documents(pdf_dir="data"):
    loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def generate_chunk_id(content, metadata):
    base = content + str(metadata)
    return hashlib.md5(base.encode()).hexdigest()


MAX_REQUEST_SIZE = 4 * 1024 * 1024  # 4 MB

def upsert_chunks(docs, embeddings, index, base_batch_size=50):
    vectors = []
    for doc in docs:
        chunk_id = generate_chunk_id(doc.page_content, doc.metadata)
        embedding = embeddings.embed_query(doc.page_content)
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                **doc.metadata,
                "text": doc.page_content
            }
        })

    def batch_generator(vectors, batch_size):
        for i in range(0, len(vectors), batch_size):
            yield vectors[i:i + batch_size]

    uploaded = 0

    for batch in batch_generator(vectors, base_batch_size):
        size_bytes = sys.getsizeof(json.dumps(batch))
        if size_bytes > MAX_REQUEST_SIZE:
            # Split big batch recursively
            half = len(batch) // 2 or 1
            for sub_batch in batch_generator(batch, half):
                index.upsert(vectors=sub_batch)
                uploaded += len(sub_batch)
                print(f"✅ Upserted sub-batch ({len(sub_batch)} vectors)")
        else:
            index.upsert(vectors=batch)
            uploaded += len(batch)
            print(f"✅ Upserted batch ({len(batch)} vectors)")

    print(f"🎉 Finished uploading {uploaded} chunks (deduplicated & batched safely)")

if __name__ == "__main__":
    print("📂 Loading documents...")
    docs = load_documents("data")

    print("✂️ Splitting documents into chunks...")
    text_chunks = split_documents(docs)

    print("🔢 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📤 Uploading chunks to Pinecone...")
    upsert_chunks(text_chunks, embeddings, index)

    print("✅ All done!")
