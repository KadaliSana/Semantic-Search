import numpy as np
import threading
import time
import pickle
import os
import logging
from typing import Dict, Tuple, Optional
from collections import OrderedDict

class SemanticCache:
    def __init__(self, threshold: float = 0.55, max_bucket_size: int = 200, ttl_seconds: int = 86400):
        self.threshold = threshold
        self.max_bucket_size = max_bucket_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
        self.store: Dict[int, OrderedDict] = {}
        self.total_entries = 0
        self.hits, self.misses = 0, 0

    def check(self, query_embedding: np.ndarray, dominant_cluster: int, category: str = None) -> Optional[Tuple[str, str, float]]:
        if dominant_cluster == -1:
            with self.lock: self.misses += 1
            return None

        with self.lock:
            if dominant_cluster not in self.store:
                self.misses += 1
                return None

            cluster_bucket = self.store[dominant_cluster]
            norm_query = query_embedding / np.linalg.norm(query_embedding)
            
            best_score, best_match = -1.0, None
            current_time = time.time()
            expired_keys = []

            for cache_key, (cached_emb, cached_result, timestamp) in cluster_bucket.items():
                cached_orig_q, cached_category = cache_key
                
                # TTL Expiration
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(cache_key)
                    continue
                
                if category != cached_category:
                    continue
                
                score = np.dot(norm_query, cached_emb)
                if score > best_score:
                    best_score = score
                    best_match = (cached_orig_q, cached_result, score, cache_key)

            for key in expired_keys:
                del cluster_bucket[key]
                self.total_entries -= 1

            if best_score >= self.threshold and best_match:
                self.hits += 1
                cache_key = best_match[3]
                # LRU update
                item = cluster_bucket.pop(cache_key)
                cluster_bucket[cache_key] = item
                return (best_match[0], best_match[1], best_score)
            
            self.misses += 1
            return None

    def add(self, query_embedding: np.ndarray, original_query: str, category: str, result: str, dominant_cluster: int):
        if dominant_cluster == -1: return
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        cache_key = (original_query, category)
        
        with self.lock:
            if dominant_cluster not in self.store:
                self.store[dominant_cluster] = OrderedDict()
            bucket = self.store[dominant_cluster]
            
            if cache_key in bucket:
                del bucket[cache_key]
                self.total_entries -= 1
                
            if len(bucket) >= self.max_bucket_size:
                bucket.popitem(last=False)
                self.total_entries -= 1
                
            bucket[cache_key] = (norm_query, result, time.time())
            self.total_entries += 1

    # Save and load previous cache results
    def save_to_disk(self, filepath: str = "semantic_cache.pkl"):
        with self.lock:
            try:
                with open(filepath, 'wb') as f: pickle.dump({'store': self.store, 'total': self.total_entries}, f)
            except Exception as e: logging.error(f"Failed to save cache: {e}")

    def load_from_disk(self, filepath: str = "semantic_cache.pkl"):
        if os.path.exists(filepath):
            with self.lock:
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        self.store = data.get('store', {})
                        self.total_entries = data.get('total', 0)
                except Exception as e: logging.error(f"Failed to load cache: {e}")

    def flush(self):
        with self.lock:
            self.store.clear()
            self.total_entries, self.hits, self.misses = 0, 0, 0

    def stats(self) -> dict:
        with self.lock:
            total = self.hits + self.misses
            return {
                "total_entries": self.total_entries, "hit_count": self.hits, 
                "miss_count": self.misses, "hit_rate": round(self.hits / total if total > 0 else 0.0, 3)
            }