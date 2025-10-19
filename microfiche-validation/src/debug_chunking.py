#!/usr/bin/env python3
"""Debug why sentences aren't grouping"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Test with simple similar sentences
test_text = """
Deep learning requires substantial resources. Neural networks need large memory. 
GPU memory is expensive and limited.

Transformers face quadratic scaling. Attention computes all token pairs. 
This limits context sizes.

Hierarchical approaches solve challenges. Multi-scale tokenization reduces tokens. 
Semantic chunking groups information.
"""

print("=" * 70)
print("DEBUGGING SEMANTIC CHUNKING")
print("=" * 70)

encoder = SentenceTransformer('all-MiniLM-L6-v2')
import re
sentences = [s.strip() for s in re.split(r'[.!?]+\s+', test_text) if s.strip() and len(s.strip()) > 10]

print(f"\nFound {len(sentences)} sentences:")
for i, s in enumerate(sentences):
    print(f"  {i}: {s[:60]}...")

embeddings = encoder.encode(sentences, convert_to_numpy=True)

print("\n" + "=" * 70)
print("SIMILARITY MATRIX")
print("=" * 70)
print("\nComparing consecutive sentences:")

for i in range(len(sentences)-1):
    similarity = np.dot(embeddings[i], embeddings[i+1]) / (
        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
    )
    print(f"\nSentence {i} → {i+1}:")
    print(f"  {sentences[i][:50]}...")
    print(f"  {sentences[i+1][:50]}...")
    print(f"  Similarity: {similarity:.4f}")
    
    if similarity > 0.70:
        print("  ✓ Would group together (> 0.70)")
    else:
        print("  ✗ Would split (< 0.70)")

# Check all pairs
print("\n" + "=" * 70)
print("ALL PAIRWISE SIMILARITIES")
print("=" * 70)

similarities = []
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[j]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
        )
        similarities.append((i, j, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[2], reverse=True)

print("\nTop 10 most similar pairs:")
for i, j, sim in similarities[:10]:
    print(f"\n  [{i}] {sentences[i][:40]}...")
    print(f"  [{j}] {sentences[j][:40]}...")
    print(f"  Similarity: {sim:.4f}")

print("\n" + "=" * 70)
