import wikipedia
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from wikipedia.exceptions import PageError, DisambiguationError


# ---------------------------------------------------------
# Wikipedia Configuration
# ---------------------------------------------------------

# Explicitly set Wikipedia language to English
wikipedia.set_lang("en")


# ---------------------------------------------------------
# Topics to Ingest
# (Domain-limited to AI / ML / GenAI)
# ---------------------------------------------------------

TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural networks",
    "Large language models",
    "Generative artificial intelligence",
    "Transformer (machine learning)",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning"
]


# ---------------------------------------------------------
# Step 1: Load Wikipedia Articles Safely
# ---------------------------------------------------------

print("🔹 Loading Wikipedia articles...")

documents = []   # Raw Wikipedia article text
metadata = []    # Metadata per article (title + URL)

for topic in TOPICS:
    try:
        # Fetch Wikipedia page without auto-suggestions
        page = wikipedia.page(topic, auto_suggest=False)

        # Store article text
        documents.append(page.content)

        # Store metadata for citation (title + link)
        metadata.append({
            "title": page.title,
            "url": page.url
        })

        print(f"✅ Loaded: {page.title}")

    except DisambiguationError as e:
        # Handle pages with multiple meanings
        try:
            # Safely pick the first suggested option
            page = wikipedia.page(e.options[0], auto_suggest=False)

            documents.append(page.content)
            metadata.append({
                "title": page.title,
                "url": page.url
            })

            print(f"⚠️ Disambiguation resolved: {page.title}")

        except Exception:
            # Skip if disambiguation cannot be resolved
            print(f"❌ Skipped (disambiguation failed): {topic}")

    except PageError:
        # Page does not exist
        print(f"❌ Page not found, skipped: {topic}")

    except Exception as e:
        # Catch-all to avoid ingestion crash
        print(f"❌ Unexpected error for {topic}: {e}")

print(f"\nTotal loaded articles: {len(documents)}")


# ---------------------------------------------------------
# Step 2: Chunk Wikipedia Text
# ---------------------------------------------------------

print("🔹 Chunking text...")

# Split long articles into smaller overlapping chunks
# This improves retrieval accuracy and LLM context handling
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []          # Individual text chunks
chunk_metadata = []  # Metadata aligned per chunk

for doc, meta in zip(documents, metadata):
    split_chunks = splitter.split_text(doc)

    for chunk in split_chunks:
        chunks.append(chunk)
        chunk_metadata.append(meta)

print(f"Total chunks created: {len(chunks)}")


# ---------------------------------------------------------
# Step 3: Create Embeddings
# ---------------------------------------------------------

print("🔹 Creating embeddings...")

# Lightweight, high-quality embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text chunks into numerical vectors
embeddings = model.encode(chunks, show_progress_bar=True)


# ---------------------------------------------------------
# Step 4: Store Embeddings in FAISS
# ---------------------------------------------------------

print("🔹 Saving FAISS index...")

# Initialize FAISS index (L2 distance)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add all embeddings to the index
index.add(np.array(embeddings))

# Ensure storage directory exists
os.makedirs("faiss_index", exist_ok=True)

# Save FAISS index to disk
faiss.write_index(index, "faiss_index/index.faiss")

# Save chunks and metadata together
with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump((chunks, chunk_metadata), f)

print("\n✅ Ingestion completed successfully!")
