import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from engine import SearchEngine
import logging

logging.getLogger().setLevel(logging.ERROR)

def run_visual_analysis():
    print("Loading Engine and Embeddings (This takes a few seconds)...")
    engine = SearchEngine()
    engine.initialize()
    
    # Extract data
    data = engine.collection.get(include=['embeddings', 'documents'])
    embeddings = np.array(data['embeddings'])
    
    # Subsample to 5000 for faster visualization rendering
    if len(embeddings) > 5000:
        np.random.seed(42)
        idx = np.random.choice(len(embeddings), 5000, replace=False)
        sample_embeddings = embeddings[idx]
        sample_docs = [data['documents'][i] for i in idx]
    else:
        sample_embeddings = embeddings
        sample_docs = data['documents']

    sns.set_theme(style="whitegrid")

    print("Bayesian Information Criterion (BIC) Curve")
    k_values = list(range(1, 100))
    bic_scores = []
    
    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type='spherical', random_state=42)
        gmm.fit(sample_embeddings)
        bic_scores.append(gmm.bic(sample_embeddings))
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, bic_scores, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("Bayesian Information Criterion (BIC) vs. Number of Clusters", fontsize=14)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("BIC Score (Lower is Better)", fontsize=12)
    plt.axvline(x=k_values[np.argmin(bic_scores)], color='r', linestyle='--', label=f'Optimal K = {k_values[np.argmin(bic_scores)]}')
    plt.legend()
    plt.tight_layout()
    plt.show()


    print("UMAP 2D Projection graph")
    # Compress 384D down to 2D for human visualization
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(sample_embeddings)
    
    # Get cluster assignments
    gmm = GaussianMixture(n_components=20, covariance_type='spherical', random_state=42)
    gmm.fit(sample_embeddings)
    labels = gmm.predict(sample_embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.title("2D UMAP Projection of Semantic Clusters (Showing Soft Boundaries)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    plt.tight_layout()
    plt.show()


    print("Cosine Similarity Distribution graph")
    
    # Simulation of a query and its similarity against the whole corpus
    test_query = "What is the best operating system for a personal computer?"
    q_emb = engine.embed_query(test_query).reshape(1, -1)
    
    similarities = cosine_similarity(q_emb, sample_embeddings)[0]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, bins=50, kde=True, color='purple')
    plt.axvline(x=0.55, color='r', linestyle='--', linewidth=2, label='Cache Hit Threshold (0.55)')
    plt.axvline(x=0.45, color='orange', linestyle='--', linewidth=2, label='Domain Outlier Threshold (0.45)')
    
    plt.title("Distribution of Cosine Similarities for a Single Query", fontsize=14)
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visual_analysis()