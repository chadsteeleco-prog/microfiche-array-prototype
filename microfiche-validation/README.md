# Microfiche Array Architecture

Hierarchical tokenization with semantic compression for efficient LLM inference.

## Key Results (H100 Validation)

- **40% memory reduction** vs baseline GPT-2
- **60% faster inference** (0.498s → 0.201s)
- **52% token reduction** (786 → 376)
- **56% compression** on technical documentation

## Architecture

Semantic-level compression using:
- Hierarchical tokenization (sentence transformers)
- 3-tier storage (GPU HBM / RAM / NVMe)
- Confidence-based expansion policies
- Compatible with standard transformers

## Validation

Tested on NVIDIA H100 80GB HBM3 with GPT-2.
Total validation cost: $9.84 (3.2 hours runtime).

## Files

- `src/` - Core implementation
- `validation/` - H100 benchmark scripts & results
- `H100_VALIDATION_SUMMARY.md` - Complete results

## Citation

Inspired by Sebastian Raschka's 2025 LLM Architecture Comparison analysis.

## License

MIT
