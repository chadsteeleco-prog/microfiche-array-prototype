#!/usr/bin/env python3
"""
H100 Compression Benchmarking - Real Performance Data
"""

import sys
sys.path.insert(0, '/workspace/src')

from semantic_chunker_v2 import ImprovedSemanticChunker
from storage_manager import TieredStorageManager
import torch
import time
import json

class H100CompressionBenchmark:
    def __init__(self):
        self.device = torch.device('cuda')
        self.results = {
            'hardware': {
                'gpu': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            },
            'benchmarks': []
        }
        
    def benchmark_text_corpus(self, name, texts, threshold=0.55):
        """Benchmark compression on a text corpus"""
        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print(f"{'='*70}")
        
        chunker = ImprovedSemanticChunker(
            similarity_threshold=threshold,
            min_chunk_size=2,
            max_chunk_size=5
        )
        
        storage = TieredStorageManager(
            storage_dir=f'./validation/results/{name}'
        )
        
        total_original = 0
        total_compressed = 0
        total_chunks = 0
        processing_times = []
        
        for i, text in enumerate(texts[:50]):  # Test 50 documents
            start = time.time()
            chunks = chunker.chunk_text(text, verbose=False)
            elapsed = time.time() - start
            
            for chunk in chunks:
                storage.store_supertoken(chunk)
                total_original += chunk['metadata']['original_length']
                total_compressed += chunk['metadata']['compressed_length']
                total_chunks += 1
            
            processing_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/50 documents...")
        
        compression_ratio = (1 - total_compressed/total_original) * 100 if total_original > 0 else 0
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        result = {
            'dataset': name,
            'documents': len(texts[:50]),
            'total_chunks': total_chunks,
            'avg_chunks_per_doc': total_chunks / 50 if total_chunks > 0 else 0,
            'compression_ratio': compression_ratio,
            'avg_processing_time': avg_time,
            'throughput_docs_per_sec': 1.0 / avg_time if avg_time > 0 else 0
        }
        
        self.results['benchmarks'].append(result)
        
        print(f"\nResults:")
        print(f"  Chunks: {total_chunks}")
        print(f"  Compression: {compression_ratio:.1f}%")
        print(f"  Avg time: {avg_time:.3f}s/doc")
        print(f"  Throughput: {result['throughput_docs_per_sec']:.1f} docs/sec")
        
        return result
    
    def run_quick_benchmark(self):
        """Run quick benchmark with sample data"""
        
        print("="*70)
        print("H100 COMPRESSION BENCHMARK - QUICK TEST")
        print("="*70)
        
        # Sample texts (realistic technical content)
        sample_texts = [
            """
            Machine learning models face significant challenges with context length scaling 
            due to quadratic attention mechanisms. The computational cost grows exponentially 
            with sequence length, limiting practical applications. In order to address this 
            limitation, researchers have proposed hierarchical tokenization approaches. These 
            methods compress semantic chunks into supertokens while maintaining detailed 
            information in separate storage tiers.
            """,
            """
            The performance benefits are substantial when processing long documents. Models 
            can maintain much larger effective context windows without proportional increases 
            in memory usage or computation time. However, the implementation requires careful 
            consideration of expansion policies to avoid computational overhead. A controller 
            must intelligently decide when to retrieve full details versus operating on 
            compressed representations.
            """,
            """
            Neural network architectures continue to evolve with transformer models leading 
            recent advances. Attention mechanisms enable modeling of long-range dependencies 
            in sequential data. Self-attention computes relationships between all positions 
            in a sequence. Multi-head attention allows the model to jointly attend to 
            information from different representation subspaces.
            """,
        ] * 17  # 51 total documents
        
        self.benchmark_text_corpus('technical_sample', sample_texts)
        
        # Save results
        output_file = './validation/results/compression_benchmark_h100.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*70}")
        
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*70)
        print("COMPRESSION BENCHMARK SUMMARY")
        print("="*70)
        print(f"\nHardware: {self.results['hardware']['gpu']}")
        print(f"Memory: {self.results['hardware']['memory_gb']:.1f} GB")
        print(f"\n{'Dataset':<20} {'Chunks':<10} {'Compression':<15} {'Throughput'}")
        print("-"*70)
        
        for bench in self.results['benchmarks']:
            print(f"{bench['dataset']:<20} "
                  f"{bench['total_chunks']:<10} "
                  f"{bench['compression_ratio']:>6.1f}% "
                  f"         {bench['throughput_docs_per_sec']:>6.1f} docs/sec")
        
        print("="*70)

if __name__ == '__main__':
    print("\n🚀 Starting H100 Compression Benchmark...")
    benchmark = H100CompressionBenchmark()
    benchmark.run_quick_benchmark()
    print("\n✅ Benchmark complete!")
