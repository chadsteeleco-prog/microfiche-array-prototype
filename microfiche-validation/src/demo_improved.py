#!/usr/bin/env python3
"""Improved demo with better compression examples"""

import sys
sys.path.insert(0, '/workspace/src')

from semantic_chunker import SemanticChunker
from storage_manager import TieredStorageManager

# More repetitive document (better compression)
technical_doc = """
Deep learning models require substantial computational resources. Neural networks consume 
large amounts of memory during training. GPU memory is particularly expensive and limited.

Transformer architectures face quadratic scaling problems. Attention mechanisms compute 
relationships between all token pairs. This quadratic complexity limits context window sizes.

Hierarchical approaches offer solutions to these challenges. Multi-scale tokenization reduces 
active token counts. Semantic chunking groups related information together.

The microfiche array architecture implements tiered storage. Fast GPU memory stores compressed 
headers. System RAM caches recently accessed data. NVMe storage holds full details.

Expansion policies determine when to retrieve details. Low confidence scores trigger expansion. 
High attention weights indicate important regions. Task-specific heuristics handle edge cases.

Performance benchmarks demonstrate effectiveness. Memory usage drops by 70-80 percent. 
Inference speed improves significantly. Answer quality remains high throughout.

Cloud infrastructure enables validation. NVIDIA containers provide standardized environments. 
Triton servers orchestrate model serving. GPUDirect accelerates data transfers.

Future work includes controller training. Reinforcement learning will optimize policies. 
Real model integration comes next. Production deployment follows validation.
"""

print("=" * 70)
print("IMPROVED DEMO - Better Compression Example")
print("=" * 70)
print()

# Try different similarity thresholds
for threshold in [0.65, 0.70, 0.75, 0.80]:
    print(f"\nTesting with similarity threshold: {threshold}")
    print("-" * 70)
    
    chunker = SemanticChunker(similarity_threshold=threshold)
    storage = TieredStorageManager(storage_dir=f'./storage/test_{int(threshold*100)}')
    
    chunks = chunker.chunk_text(technical_doc, verbose=False)
    
    # Store all
    for chunk in chunks:
        storage.store_supertoken(chunk)
    
    # Calculate metrics
    total_original = sum(c['metadata']['original_length'] for c in chunks)
    total_compressed = sum(c['metadata']['compressed_length'] for c in chunks)
    compression = (1 - total_compressed/total_original) * 100
    avg_confidence = sum(c['confidence'] for c in chunks) / len(chunks)
    
    print(f"  Supertokens created: {len(chunks)}")
    print(f"  Avg sentences/chunk: {sum(c['metadata']['sentence_count'] for c in chunks) / len(chunks):.1f}")
    print(f"  Compression ratio: {compression:.1f}%")
    print(f"  Average confidence: {avg_confidence:.3f}")
    
    # Count low confidence (would expand)
    low_conf = sum(1 for c in chunks if c['confidence'] < 0.75)
    print(f"  Would expand: {low_conf}/{len(chunks)} ({low_conf/len(chunks)*100:.1f}%)")

print("\n" + "=" * 70)
print("✓ Threshold comparison complete!")
print("=" * 70)
