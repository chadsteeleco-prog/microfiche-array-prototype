#!/usr/bin/env python3
"""
Improved Semantic Chunker with better multi-sentence grouping
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import hashlib
import re
import time

class ImprovedSemanticChunker:
    """
    Better chunking: considers paragraph structure + semantic similarity
    """
    
    def __init__(self, similarity_threshold: float = 0.60, 
                 min_chunk_size: int = 2, max_chunk_size: int = 5):
        print("Initializing Improved Semantic Chunker...")
        start = time.time()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        print(f"  ✓ Ready in {time.time() - start:.2f}s")
        print(f"  Threshold: {similarity_threshold}, Chunk size: {min_chunk_size}-{max_chunk_size}")
        
    def chunk_text(self, text: str, verbose: bool = True) -> List[Dict]:
        if verbose:
            print(f"\nProcessing text ({len(text)} chars)...")
        
        # First split by paragraphs (major semantic boundaries)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        all_chunks = []
        for para in paragraphs:
            # Within each paragraph, chunk by sentences
            sentences = self._split_sentences(para)
            if len(sentences) == 0:
                continue
                
            if verbose:
                print(f"  Paragraph: {len(sentences)} sentences")
            
            # Encode all sentences in paragraph
            embeddings = self.encoder.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
            
            # Group with forced minimum chunk sizes
            chunks = self._group_with_min_size(sentences, embeddings)
            all_chunks.extend(chunks)
        
        if verbose:
            print(f"  ✓ Created {len(all_chunks)} supertokens")
            self._print_stats(all_chunks)
        
        return all_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _group_with_min_size(self, sentences: List[str], embeddings: np.ndarray) -> List[Dict]:
        """Group sentences, forcing minimum chunk sizes"""
        chunks = []
        i = 0
        
        while i < len(sentences):
            current_chunk = [sentences[i]]
            current_embeddings = [embeddings[i]]
            i += 1
            
            # Force at least min_chunk_size sentences
            while len(current_chunk) < self.min_chunk_size and i < len(sentences):
                current_chunk.append(sentences[i])
                current_embeddings.append(embeddings[i])
                i += 1
            
            # Try to add more if similar and under max size
            while len(current_chunk) < self.max_chunk_size and i < len(sentences):
                chunk_centroid = np.mean(current_embeddings, axis=0)
                similarity = np.dot(chunk_centroid, embeddings[i]) / (
                    np.linalg.norm(chunk_centroid) * np.linalg.norm(embeddings[i])
                )
                
                if similarity > self.similarity_threshold:
                    current_chunk.append(sentences[i])
                    current_embeddings.append(embeddings[i])
                    i += 1
                else:
                    break
            
            # Create supertoken
            chunk_embedding = np.mean(current_embeddings, axis=0)
            chunks.append(self._create_supertoken(current_chunk, chunk_embedding, current_embeddings))
        
        return chunks
    
    def _create_supertoken(self, sentences: List[str], 
                          chunk_embedding: np.ndarray,
                          sentence_embeddings: List[np.ndarray]) -> Dict:
        full_text = ' '.join(sentences)
        chunk_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
        
        # Better confidence calculation
        if len(sentence_embeddings) > 1:
            # Variance from chunk centroid
            distances = [np.linalg.norm(e - chunk_embedding) for e in sentence_embeddings]
            variance = np.var(distances)
            confidence = 1.0 / (1.0 + variance)
        else:
            # Single sentence = perfect confidence
            confidence = 1.0
        
        # Compressed summary: first 100 chars
        summary = full_text[:100] + '...' if len(full_text) > 100 else full_text
        
        return {
            'id': f'ST_{chunk_hash}',
            'summary': summary,
            'embedding': chunk_embedding,
            'confidence': float(confidence),
            'full_text': full_text,
            'metadata': {
                'sentence_count': len(sentences),
                'original_length': len(full_text),
                'compressed_length': len(summary),
                'compression_ratio': 1.0 - (len(summary) / len(full_text)) if len(full_text) > 0 else 0
            }
        }
    
    def _print_stats(self, chunks: List[Dict]):
        total_original = sum(c['metadata']['original_length'] for c in chunks)
        total_compressed = sum(c['metadata']['compressed_length'] for c in chunks)
        avg_confidence = np.mean([c['confidence'] for c in chunks])
        avg_sentences = np.mean([c['metadata']['sentence_count'] for c in chunks])
        
        print(f"\n  Statistics:")
        print(f"    Avg sentences/chunk: {avg_sentences:.1f}")
        print(f"    Total original: {total_original} chars")
        print(f"    Total compressed: {total_compressed} chars")
        print(f"    Overall compression: {(1 - total_compressed/total_original)*100:.1f}%")
        print(f"    Average confidence: {avg_confidence:.3f}")


# Quick test
if __name__ == '__main__':
    doc = """
Deep learning models require substantial computational resources. Neural networks consume 
large amounts of memory during training. GPU memory is particularly expensive and limited.

Transformer architectures face quadratic scaling problems. Attention mechanisms compute 
relationships between all token pairs. This quadratic complexity limits context window sizes.

Hierarchical approaches offer solutions to these challenges. Multi-scale tokenization reduces 
active token counts. Semantic chunking groups related information together.
    """
    
    chunker = ImprovedSemanticChunker(similarity_threshold=0.60, min_chunk_size=2, max_chunk_size=4)
    chunks = chunker.chunk_text(doc)
    
    print("\n" + "=" * 70)
    print("CHUNKS CREATED:")
    for i, c in enumerate(chunks, 1):
        print(f"\n{i}. {c['id']} ({c['metadata']['sentence_count']} sentences, conf={c['confidence']:.3f})")
        print(f"   {c['summary']}")
