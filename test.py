import requests
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
import logging

logging.getLogger().setLevel(logging.ERROR)
BASE_URL = "http://localhost:8000"

def get_realistic_queries(pool_size=200):
    print("Fetching unseen Test Split from 20 Newsgroups...")
    # Fetch the test set, which your database has never seen
    test_data = fetch_20newsgroups(subset='test', remove=('footers', 'quotes'))
    
    queries = []
    for text in test_data.data:
        match = re.search(r'^Subject:\s*(.*)', text, re.MULTILINE)
        if match:
            clean_q = re.sub(r'^(Re:\s*|Fwd:\s*)+', '', match.group(1).strip(), flags=re.IGNORECASE)
            if len(clean_q) > 15: # Only keep meaningful queries
                queries.append(clean_q)
        if len(queries) >= pool_size:
            break
            
    print(f"Extracted {len(queries)} unique human queries from Subject lines.")
    return queries

def generate_zipfian_traffic(queries, num_requests=1000, alpha=1.5):
    """
    Generates a realistic traffic stream using a Zipfian distribution.
    alpha=1.5 simulates a typical web workload where a few items are highly popular.
    """
    print(f"Generating Zipfian traffic stream of {num_requests} requests...")
    ranks = np.arange(1, len(queries) + 1)
    probabilities = 1.0 / (ranks ** alpha)
    probabilities /= probabilities.sum() # Normalize to 1.0
    
    # Draw indices based on the Zipfian probabilities
    indices = np.random.choice(len(queries), size=num_requests, p=probabilities)
    return [queries[i] for i in indices]

def run_benchmark():
    # 1. Setup the data
    unique_queries = get_realistic_queries(pool_size=100)
    traffic_stream = generate_zipfian_traffic(unique_queries, num_requests=500)
    
    print("\nFlushing cache for a clean benchmark run...")
    try:
        requests.delete(f"{BASE_URL}/cache")
    except Exception:
        print("Error: Cannot connect to FastAPI server. Is it running?")
        return

    results = []
    cumulative_hits = 0
    
    print("Firing benchmark traffic at the API...")
    # 2. Execute the traffic stream
    for i, q in enumerate(traffic_stream):
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/query", json={"query": q}).json()
        latency = (time.time() - start_time) * 1000 # ms
        
        cache_hit = response.get("cache_hit", False)
        if cache_hit:
            cumulative_hits += 1
            
        hit_rate = cumulative_hits / (i + 1)
        
        results.append({
            "Request Number": i + 1,
            "Latency (ms)": latency,
            "Cache Hit": cache_hit,
            "Cumulative Hit Rate": hit_rate
        })
        
        # Progress every 50 requests
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/500 requests | Current Hit Rate: {hit_rate*100:.1f}%")

    df = pd.DataFrame(results)
    plot_benchmark_metrics(df)

def plot_benchmark_metrics(df):
    print("\nGenerating Benchmark Dashboards...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(18, 5))
    fig.suptitle("Zipfian Traffic Load - Cache Benchmark", fontsize=16, fontweight='bold')

    # Hit Rate Over Time 
    sns.lineplot(data=df, x="Request Number", y="Cumulative Hit Rate", color="blue", ax=axes, linewidth=2)
    axes.set_title("Cache Hit Rate over Time (Warm-up Curve)", fontsize=14)
    axes.set_ylabel("Hit Rate (%)", fontsize=12)
    axes.set_ylim(0, 1.0)
    
    final_rate = df["Cumulative Hit Rate"].iloc[-1]
    axes.axhline(final_rate, color='r', linestyle='--', alpha=0.6)
    axes.text(250, final_rate + 0.05, f"Stable Hit Rate: {final_rate*100:.1f}%", color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()