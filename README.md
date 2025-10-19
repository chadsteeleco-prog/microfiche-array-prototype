# microfiche-array-prototype
**Hierarchical tokenization with semantic compression for efficient LLM inference**
# Microfiche Array Architecture

**Hierarchical tokenization with semantic compression for efficient LLM inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Results (H100 Validation)

Validated on **NVIDIA H100 80GB HBM3** with GPT-2:

- **40% memory reduction** (0.152 GB → 0.091 GB)
- **60% faster inference** (0.498s → 0.201s)  
- **52% token reduction** (786 → 376 tokens)
- **56% compression** on technical documentation
- **$9.84 total validation cost** (3.2 hours H100 runtime)

## 💡 Core Innovation

While existing optimizations (MLA, GQA) compress at the attention/tensor level, 
this architecture compresses at the **semantic level**—grouping related content 
into hierarchical "supertokens" before it reaches the attention mechanism.

## 🏗️ Architecture

### Three-Tier Storage Hierarchy
```
Tier 1 (GPU HBM)  → Supertoken headers (fast, limited)
Tier 2 (System RAM) → Expanded cache (medium)
Tier 3 (NVMe)      → Full details (large, slower)
```

### Components
- **Semantic Chunker:** Groups sentences by semantic similarity
- **Storage Manager:** Manages 3-tier memory hierarchy with LRU eviction
- **Expansion Controller:** Confidence-based policies for detail retrieval

## 📊 Benchmark Results

### Compression Performance
- Dataset: Technical documentation (50 docs)
- Throughput: **111.8 docs/second**
- Processing: **0.009s per document**
- Compression: **56.0%**

### Model Integration (GPT-2)
| Metric | Baseline | Microfiche | Improvement |
|--------|----------|------------|-------------|
| Input Tokens | 786 | 376 | **-52%** |
| Memory Used | 0.152 GB | 0.091 GB | **-40%** |
| Inference Time | 0.498s | 0.201s | **-60%** |
| Text Compression | - | 45.8% | **46%** |

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/microfiche-array-architecture.git
cd microfiche-array-architecture

# Install dependencies
pip install sentence-transformers transformers torch

# Run demo
python src/final_demo.py
```

## 📁 Repository Structure
```
├── src/
│   ├── semantic_chunker_v2.py      # Hierarchical tokenization
│   ├── storage_manager.py          # 3-tier storage system
│   ├── final_demo.py               # End-to-end demonstration
│   └── prototype_demo.py           # Integration test
├── validation/
│   ├── benchmark_compression_h100.py   # H100 compression tests
│   ├── model_integration_h100.py       # GPT-2 integration
│   └── results/                        # JSON benchmark data
├── H100_VALIDATION_SUMMARY.md      # Complete validation report
└── README.md
```

## 🔬 Technical Details

### Semantic Chunking
Uses sentence transformers (`all-MiniLM-L6-v2`) to:
1. Encode sentences into embeddings
2. Group by cosine similarity (threshold: 0.55-0.60)
3. Create supertokens with confidence scores

### Storage Management
- **Tier 1:** 20 supertoken capacity, LRU eviction
- **Tier 2:** 50 expanded chunk cache
- **Tier 3:** Unlimited persistent storage

### Expansion Policy
Expands supertokens when:
- Confidence score < 0.75
- High attention weight (> 0.15)
- Task-specific heuristics (code, math, legal)

## 📄 Citation

Inspired by Sebastian Raschka's "The Big LLM Architecture Comparison" (2025).

If you use this work, please cite:
```bibtex
@software{microfiche_array_2025,
  author = {Steele, Chad},
  title = {Microfiche Array Architecture},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/microfiche-array-architecture}
}
```

## 🛠️ Hardware Requirements

**For prototyping:**
- CPU: Any modern processor
- RAM: 8GB+
- No GPU required

**For validation:**
- GPU: NVIDIA H100 (or A100)
- VRAM: 40GB+
- Storage: 500GB+ NVMe

## 📊 Validation Methodology

Complete validation performed on:
- **Hardware:** NVIDIA H100 80GB HBM3
- **Framework:** PyTorch 2.5.0, CUDA 12.6
- **Model:** GPT-2 (124M parameters)
- **Duration:** 3.2 hours
