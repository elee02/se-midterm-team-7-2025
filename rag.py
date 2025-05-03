import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self):
        # Choose GPU if available, else CPU
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing embeddings on device: {device_str}")
        self.embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device_str)
        logger.info(f"SentenceTransformer loaded onto device: {self.embedder.device}")
        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = {}
        self.index_file = "faiss_index/index.faiss"
        self.metadata_file = "faiss_index/metadata.pkl"
        os.makedirs("faiss_index", exist_ok=True)
        self.embeddings = {}  # map chunk_id → normalized embedding

        if os.path.exists(self.index_file):
            loaded_index = faiss.read_index(self.index_file)
            if loaded_index.d != embedding_dim:
                logger.warning("Index dimension mismatch! Recreating index.")
                self.index = faiss.IndexFlatL2(embedding_dim)
                self.metadata = {}
            else:
                self.index = loaded_index
                with open(self.metadata_file, "rb") as f:
                    self.metadata = pickle.load(f)

    async def embed_document(self, file_id: str, text: str):
        logger.info(f"Embedding document file_id={file_id}, length={len(text)}")
        # Remove textwrap usage and split text by sentences:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_size = 500  # Increased chunk size

        for sentence in sentences:
            if current_length + len(sentence) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.debug(f"Generated {len(chunks)} chunks for file_id={file_id}")
        if not chunks:
            logger.warning(f"Empty document provided for file_id {file_id}")
            raise ValueError("Document contains no text to embed.")

        # Log chunk stats
        logger.info(f"file_id={file_id}: generated {len(chunks)} chunks")
        lengths = [len(c) for c in chunks]
        logger.debug(
            f"file_id={file_id}: chunk sizes → min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}"
        )

        # add overlap: keep last 3 sentences in next chunk
        overlap_sents = 3
        chunks_with_overlap = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev = chunks_with_overlap[-1].split(' ')
                overlap = ' '.join(prev[-overlap_sents:])
                chunk = overlap + ' ' + chunk
            chunks_with_overlap.append(chunk)
        chunks = chunks_with_overlap

        # Ensure embeddings are generated on CPU
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True) # convert_to_numpy might help ensure CPU usage
        # normalize for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        logger.debug(f"Embeddings shape: {embeddings.shape}")
        ids = np.array([i for i in range(len(self.metadata), len(self.metadata) + len(chunks))])
        # index normalized vectors so L2 on them ≃ cosine
        self.index.add(normalized.astype("float32"))

        # Log new index size
        logger.info(f"After embedding file_id={file_id}, index contains {self.index.ntotal} vectors")

        # store for direct cosine lookup
        for i, idx in enumerate(ids):
            self.embeddings[int(idx)] = normalized[i]

        for i, chunk in enumerate(chunks):
            logger.debug(f"Indexing chunk {i} (length={len(chunk)}) for file_id={file_id}")
            self.metadata[ids[i]] = {"file_id": file_id, "text": chunk}

        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Completed embedding for file_id={file_id}")

    async def query_document(self, question: str, file_id: str) -> list:
        # Ensure query embedding is generated on CPU
        raw_q = self.embedder.encode([question], convert_to_numpy=True)[0]
        # normalize query
        q_norm = raw_q / (np.linalg.norm(raw_q) + 1e-10)

        k = 20  # increased recall
        distances, indices = self.index.search(np.array([q_norm.astype("float32")]), k=k)

        # Log raw neighbors
        logger.info(
            f"file_id={file_id}: FAISS returned {len(indices[0])} neighbors; distances={distances[0].tolist()}"
        )

        # collect only same-file candidates - also check if index exists in embeddings
        candidates = []
        for idx in indices[0]:
            idx = int(idx)
            # Only include if it belongs to the requested file and exists in embeddings
            if idx in self.metadata and self.metadata[idx]["file_id"] == file_id and idx in self.embeddings:
                candidates.append(idx)
        
        logger.info(f"file_id={file_id}: {len(candidates)} valid candidates after filtering")
        
        if not candidates:
            logger.warning(f"No valid candidates found for file_id={file_id}")
            return []

        # rank by true cosine similarity
        try:
            scored = []
            for idx in candidates:
                # Safely compute similarity with checks
                if idx in self.embeddings:
                    sim = float(np.dot(self.embeddings[idx], q_norm))
                    scored.append((idx, sim))
            
            # Sort by similarity (highest first)
            scored.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"file_id={file_id}: top similarities → {[sim for _, sim in scored[:5]]}")
            
            # Always return top 3 results regardless of threshold
            picks = [idx for idx, _ in scored[:3]]
            logger.info(f"file_id={file_id}: selected {len(picks)} passages")
            
            # Return the text content
            return [self.metadata[idx]["text"] for idx in picks]
        except Exception as e:
            logger.error(f"Error during similarity calculation: {str(e)}")
            # If anything goes wrong, just return the top 3 candidates based on FAISS distances
            top_indices = [int(idx) for idx in indices[0][:3] if int(idx) in self.metadata]
            if top_indices:
                return [self.metadata[idx]["text"] for idx in top_indices]
            return []