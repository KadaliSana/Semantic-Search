import os
import logging
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.datasets import load_files
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO)

class SearchEngine:
    def __init__(self):
        self.is_ready = False
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.cross_encoder = CrossEncoder('cross-encoder/stsb-TinyBERT-L-4')
        
        self.db_path = "./chroma_data"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="20_newsgroups_semantic",
            metadata={"hnsw:space": "cosine"}
        )

    def initialize(self):
        """Runs in a background thread to prevent API blocking on boot."""
        if self.collection.count() == 0:
            self._populate_database()
        self._fit_clusters()
        
        self.is_ready = True
        logging.info("Search Engine is fully initialized and ready to serve requests.")

    def _populate_database(self):
        dataset = load_files(container_path="./20_newsgroups", encoding="utf-8", decode_error="replace")
        
        documents, metadatas = [], []
        
        for i, doc in enumerate(dataset.data):
            parts = doc.split('\n\n', 1)
            clean_doc = parts[1] if len(parts) > 1 else doc
            
            if len(clean_doc.strip()) > 50:
                documents.append(clean_doc)
                category_name = dataset.target_names[dataset.target[i]]
                metadatas.append({"category": category_name})
                
        total_docs = len(documents)
        logging.info(f"Found {total_docs} valid documents.")
        
        batch_size = 5000
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_docs))]
            
            logging.info(f"Embedding batch {i // batch_size + 1}...")
            batch_embeddings = self.model.encode(batch_docs, convert_to_numpy=True).tolist()
            
            self.collection.add(
                ids=batch_ids, documents=batch_docs, embeddings=batch_embeddings, metadatas=batch_meta
            )
        logging.info("Database fully populated!")

    def _fit_clusters(self):
        logging.info("Fitting Spherical GMM...")
        all_data = self.collection.get(include=['embeddings'])
        embeddings = np.array(all_data['embeddings'])
        
        self.n_clusters = 45 #From Bayesian Information Curve Optimal number of clusters is 45 (Check analysis.py)
        self.gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='spherical', random_state=42)
        self.gmm.fit(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]

    def get_cluster_info(self, query_embedding: np.ndarray) -> tuple[int, list[float]]:
        distribution = self.gmm.predict_proba([query_embedding])[0]
        dominant_cluster = int(np.argmax(distribution))
        confidence = distribution[dominant_cluster]
        
        if confidence < 0.15:
            dominant_cluster = -1
        return dominant_cluster, distribution.tolist()

    def search_database(self, query_embedding: np.ndarray, category: str = None) -> tuple[str, float]:
        query_args = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": 1,
            "include": ['documents', 'distances', 'metadatas']
        }
        
        if category:
            query_args["where"] = {"category": category}

        results = self.collection.query(**query_args)
        
        if results['documents'] and results['documents'][0]:
            doc_text = results['documents'][0][0]
            similarity = 1.0 - results['distances'][0][0]
            return doc_text, similarity
            
        return "No relevant documents found.", 0.0

    def verify_cache_hit(self, original_query: str, cached_query: str) -> bool:
        """Cross-Encoder catches logical contradictions the Bi-Encoder misses."""
        score = self.cross_encoder.predict([original_query, cached_query])
        return bool(score > 0.0)