import requests
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch

def fetch_arxiv_papers(query="machine learning", max_results=10):
    """
    Fetch recent papers from arXiv using dummy data (simulated response).
    In a real implementation, you would parse the XML response.
    """
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
    except Exception as e:
        print("Request failed, using dummy data. Error:", e)
    
    papers = []
    for i in range(max_results):
        papers.append({
            "title": f"Paper {i}",
            "summary": f"This is a sample summary about machine learning technique {i}.",
            "published": datetime.now().strftime("%Y-%m-%d")
        })
    print("Fetched papers from arXiv (dummy):", papers)
    return papers

def update_knowledge_base(papers):
    """
    Process and cluster retrieved paper summaries.
    Returns a dictionary mapping cluster label to list of papers.
    """
    texts = [paper["summary"] for paper in papers]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    num_clusters = min(3, len(papers))
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    expert_kb = {}
    for idx, label in enumerate(clusters):
        expert_kb.setdefault(label, []).append(papers[idx])
    print("Updated Expert Knowledge Base (Clusters):", expert_kb)
    return expert_kb

def generate_synthetic_regression_data(n=100):
    """
    Generate synthetic regression data: y = 3*x + noise.
    Returns torch tensors.
    """
    X = np.random.rand(n, 1).astype(np.float32)
    noise = np.random.randn(n, 1).astype(np.float32) * 0.1
    y = 3 * X + noise
    return torch.tensor(X), torch.tensor(y)

def generate_buggy_code():
    """
    Return a string with an intentionally buggy code snippet
    to simulate a linear regression task using PyTorch.
    The bug is a wrong input dimension for nn.Linear.
    """
    buggy_code = """
import torch
import torch.nn as nn
import torch.optim as optim

def generate_data(n=50):
    import numpy as np
    X = np.random.rand(n, 1).astype('float32')
    y = 3 * X.squeeze() + np.random.randn(n).astype('float32') * 0.1
    return torch.tensor(X), torch.tensor(y)

model = nn.Linear(2, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
X, y = generate_data()
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
print("Final loss:", loss.item())
"""
    return buggy_code
