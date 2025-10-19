#!/usr/bin/env python3
"""
H100 Model Integration - GPT-2 with Microfiche Array
"""

import sys
sys.path.insert(0, '/workspace/src')

from semantic_chunker_v2 import ImprovedSemanticChunker
from storage_manager import TieredStorageManager
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import json

class H100ModelIntegration:
    def __init__(self):
        self.device = torch.device('cuda')
        print("Loading GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.chunker = ImprovedSemanticChunker()
        self.storage = TieredStorageManager(storage_dir='./validation/results/model_test')
        
    def test_baseline(self, text):
        """Baseline: standard GPT-2 processing"""
        print("\n" + "="*70)
        print("BASELINE TEST (Standard GPT-2)")
        print("="*70)
        
        # Reset GPU memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        start_mem = torch.cuda.memory_allocated() / 1e9
        
        # Tokenize and process
        start_time = time.time()
        inputs = self.tokenizer(text, return_tensors='pt', 
                               padding=True, truncation=True, 
                               max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result = {
            'method': 'baseline',
            'input_tokens': inputs.input_ids.shape[1],
            'memory_used_gb': peak_mem - start_mem,
            'peak_memory_gb': peak_mem,
            'time_seconds': elapsed,
            'output_length': len(response)
        }
        
        print(f"Input tokens: {result['input_tokens']}")
        print(f"Memory used: {result['memory_used_gb']:.3f} GB")
        print(f"Peak memory: {result['peak_memory_gb']:.3f} GB")
        print(f"Time: {result['time_seconds']:.3f}s")
        print(f"Output: {response[:100]}...")
        
        return result
    
    def test_microfiche(self, text):
        """Microfiche: with semantic compression"""
        print("\n" + "="*70)
        print("MICROFICHE TEST (With Compression)")
        print("="*70)
        
        # Chunk text first
        chunks = self.chunker.chunk_text(text, verbose=False)
        print(f"Created {len(chunks)} supertokens")
        
        # Store chunks
        for chunk in chunks:
            self.storage.store_supertoken(chunk)
        
        # Process compressed representation
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        start_mem = torch.cuda.memory_allocated() / 1e9
        
        # Use compressed summaries
        compressed_text = ' '.join([c['summary'] for c in chunks])
        print(f"Compressed length: {len(compressed_text)} chars (vs {len(text)} original)")
        
        start_time = time.time()
        inputs = self.tokenizer(compressed_text, return_tensors='pt',
                               padding=True, truncation=True,
                               max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate compression stats
        compression_ratio = (1 - len(compressed_text)/len(text)) * 100
        
        result = {
            'method': 'microfiche',
            'supertokens': len(chunks),
            'compression_ratio': compression_ratio,
            'input_tokens': inputs.input_ids.shape[1],
            'memory_used_gb': peak_mem - start_mem,
            'peak_memory_gb': peak_mem,
            'time_seconds': elapsed,
            'output_length': len(response)
        }
        
        print(f"Supertokens: {result['supertokens']}")
        print(f"Compression: {result['compression_ratio']:.1f}%")
        print(f"Input tokens: {result['input_tokens']}")
        print(f"Memory used: {result['memory_used_gb']:.3f} GB")
        print(f"Peak memory: {result['peak_memory_gb']:.3f} GB")
        print(f"Time: {result['time_seconds']:.3f}s")
        print(f"Output: {response[:100]}...")
        
        return result
    
    def run_comparison(self):
        """Run complete comparison"""
        test_text = """
        The field of artificial intelligence has made remarkable progress in recent years.
        Large language models like GPT demonstrate unprecedented capabilities in natural
        language understanding and generation. However, these models face significant
        challenges with memory efficiency and context length limitations. The quadratic
        scaling of attention mechanisms creates bottlenecks for processing long documents.
        
        Researchers are exploring hierarchical approaches to address these limitations.
        Multi-scale tokenization and semantic chunking offer promising solutions. By
        organizing information into tiered structures, models can maintain larger effective
        context windows. This enables processing of book-length documents without
        proportional increases in computational cost.
        
        The microfiche array architecture implements these principles through a three-tier
        storage hierarchy. Fast GPU memory stores compressed headers. System RAM caches
        recently accessed data. NVMe storage holds full details. Expansion policies determine
        when to retrieve complete information based on confidence scores and attention patterns.
        
        Performance benchmarks demonstrate the effectiveness of this approach. Memory usage
        drops significantly while maintaining answer quality. Inference speed improves through
        reduced token processing. The system enables practical deployment of models with
        extended context capabilities at reduced computational cost.
        """ * 2  # Repeat for longer context
        
        print("="*70)
        print("H100 MODEL INTEGRATION TEST")
        print("="*70)
        print(f"\nTest document: {len(test_text)} characters")
        
        baseline_result = self.test_baseline(test_text)
        microfiche_result = self.test_microfiche(test_text)
        
        # Calculate improvements
        memory_reduction = ((baseline_result['memory_used_gb'] - microfiche_result['memory_used_gb']) 
                          / baseline_result['memory_used_gb'] * 100) if baseline_result['memory_used_gb'] > 0 else 0
        
        time_change = ((microfiche_result['time_seconds'] - baseline_result['time_seconds'])
                      / baseline_result['time_seconds'] * 100) if baseline_result['time_seconds'] > 0 else 0
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\nCompression: {microfiche_result['compression_ratio']:.1f}%")
        print(f"Memory reduction: {memory_reduction:.1f}%")
        print(f"Time change: {time_change:+.1f}%")
        print(f"Token reduction: {baseline_result['input_tokens']} → {microfiche_result['input_tokens']}")
        
        # Save results
        results = {
            'hardware': {
                'gpu': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            },
            'baseline': baseline_result,
            'microfiche': microfiche_result,
            'improvements': {
                'compression_ratio': microfiche_result['compression_ratio'],
                'memory_reduction_percent': memory_reduction,
                'time_change_percent': time_change,
                'token_reduction': baseline_result['input_tokens'] - microfiche_result['input_tokens']
            }
        }
        
        output_file = './validation/results/model_integration_h100.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print("="*70)
        
        return results

if __name__ == '__main__':
    print("\n🚀 Starting H100 Model Integration Test...")
    test = H100ModelIntegration()
    test.run_comparison()
    print("\n✅ Model integration test complete!")
