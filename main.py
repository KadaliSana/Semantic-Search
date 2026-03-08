from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging
from engine import SearchEngine
from cache import SemanticCache

app = FastAPI(title="Production Semantic Cache API")

engine = SearchEngine()
cache = SemanticCache(threshold=0.55, max_bucket_size=200)
CACHE_FILE = "semantic_cache.pkl"

@app.on_event("startup")
async def startup_event():
    cache.load_from_disk(CACHE_FILE)
    asyncio.create_task(asyncio.to_thread(engine.initialize))

@app.on_event("shutdown")
async def shutdown_event():
    cache.save_to_disk(CACHE_FILE)

class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None  # User can filter by domain (e.g. 'comp.graphics')

@app.post("/query")
async def process_query(req: QueryRequest):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model is warming up. Please try again in a few moments.")
        
    q_emb = await asyncio.to_thread(engine.embed_query, req.query)
    dominant_cluster, cluster_distribution = await asyncio.to_thread(engine.get_cluster_info, q_emb)
    cache_result = cache.check(q_emb, dominant_cluster, req.category)
    
    if cache_result:
        matched_query, result_text, score = cache_result
        is_valid = await asyncio.to_thread(engine.verify_cache_hit, req.query, matched_query)
        
        if is_valid:
            return {
                "query": req.query, "category_filter": req.category, "cache_hit": True,
                "matched_query": matched_query, "similarity_score": round(float(score), 3),
                "result": result_text, "dominant_cluster": dominant_cluster
            }
        else:
            logging.warning(f"Cross-Encoder rejected cache hit for: '{req.query}' vs '{matched_query}'")

    result_text, db_similarity = await asyncio.to_thread(engine.search_database, q_emb, req.category)
    
    if db_similarity < 0.45:
        result_text = "Out of Domain: I couldn't find anything relevant in the dataset."
        dominant_cluster = -1
        
    cache.add(q_emb, req.query, req.category, result_text, dominant_cluster)
    
    return {
        "query": req.query, "category_filter": req.category, "cache_hit": False,
        "result": result_text, "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_cache_stats():
    return cache.stats()

@app.delete("/cache")
async def flush_cache():
    cache.flush()
    return {"status": "Cache flushed successfully"}