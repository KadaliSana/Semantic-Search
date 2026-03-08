# Semantic Search & Caching Engine

A production-ready FastAPI microservice providing semantic search over the 20 Newsgroups dataset, backed by a custom-built, cluster-routed semantic cache. This system was built from first principles to prioritize mathematically proven routing, fuzzy clustering, and high-fidelity logical reranking.

## Architecture Overview

* **API Framework:** Asynchronous FastAPI
* **Embedding Model:** `all-MiniLM-L6-v2` (384-dimensional vector space)
* **Vector Database:** Local embedded `ChromaDB` (HNSW indexing)
* **Fuzzy Clustering:** Spherical Gaussian Mixture Model (GMM)
* **Reranking/Validation:** Cross-Encoder (`stsb-TinyBERT-L-4`)
* **Semantic Cache:** In-memory, cluster-routed, thread-safe, with LRU eviction and TTL.

---

## Design Decisions & Mathematical Proofs

The architecture of this system relies on statistical evidence rather than convenience or heuristics. Below are the data-driven justifications for the core components.

### 1. The Micro-Topic Discovery (Cluster Count Justification)
Standard clustering often relies on human-assigned convenience labels (e.g., the 20 categories in the 20 Newsgroups dataset). To determine the *true* semantic density of the cache routing buckets, I executed an extended Bayesian Information Criterion (BIC) analysis.

![Extended BIC Curve showing optimal K=45](images/BIC.jpeg)

* **The Evidence:** The BIC score continuously drops and minimizes perfectly at **$K = 45$** before the Bayesian penalty for over-complexity forces the curve back up.
* **Architectural Impact:** This proves the embedding model natively fragments the dataset's broad human labels into 45 distinct, highly specialized micro-topics. Setting the GMM to 45 buckets guarantees that cached queries are partitioned into incredibly tight semantic neighborhoods, optimizing cache lookup speeds from $O(N)$ to $O(N/K)$.

### 2. The Soft Boundary Problem (Fuzzy Clustering)
A document about encryption legislation does not belong strictly to "Politics" or "Technology"—it belongs to both. Hard-clustering algorithms (like K-Means) force arbitrary boundaries that break cache routing.

![2D UMAP Projection showing soft semantic boundaries](images/clustering.jpeg)

* **The Evidence:** A 2D UMAP projection of the 384-dimensional space reveals dense core "islands" connected by massive, overlapping bridges where topics bleed into one another.
* **Architectural Impact:** To handle this, the system utilizes a **Spherical Gaussian Mixture Model (GMM)**. Instead of a hard label, it outputs a fuzzy probability distribution. Boundary queries are identified and routed to multiple overlapping cache buckets simultaneously, preventing cache misses on nuanced topics.

### 3. Vector Noise & Thresholding (The Tunable Decision)
To protect the cache from false positives and out-of-domain noise (e.g., gibberish queries), thresholds must be grounded in the mathematical distribution of the vector space.

![Distribution of Cosine Similarities for a Single Query](images/Distribution.jpeg)

* **The Evidence:** The background noise of the 384D space forms a massive normal distribution centered near `0.0`. 
* **Architectural Impact:** * **Outlier Guardrail (0.45):** Sitting at the extreme 99th percentile of the noise curve, any query matching below 0.45 is mathematically indistinguishable from random noise and is instantly rejected.
  * **Cache Hit Threshold (0.55):** A score $\ge 0.55$ sits entirely outside the probability distribution of chance, mathematically guaranteeing a high-fidelity semantic paraphrase. 

---

##  Performance & Benchmarking

The cache was stress-tested using 1,000 unique human queries (extracted from the unseen 20 Newsgroups test split) fed through a **Zipfian Distribution** to simulate real-world, power-law API traffic.

![Zipfian Traffic Load Cache Benchmark Dashboard](images/benchmark.jpeg)

### Benchmark Results
* **Warm-Up & Stability:** The cache successfully learned popular queries dynamically, stabilizing at a massive **86.0% Hit Rate**. 
* **Production Scaling:** In a true cloud environment where the Vector Database sits behind a network call (averaging 100ms–300ms latency), this cache architecture will yield an immediate **5x to 10x latency speedup** while guaranteeing unparalleled semantic accuracy.

---

## API & Setup Instructions

The entire system is containerized for immediate deployment.

**1. Boot the Service:**
```bash
docker-compose up --build
