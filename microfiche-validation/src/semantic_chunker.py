#!/usr/bin/env python3
"""
Microfiche Array Prototype - Semantic Chunker
Day 1 Component: Convert text into semantic supertokens
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import hashlib
import re
import time

class SemanticChunker:
    """
    Converts text into hierarchical supertokens based on semantic similarity.
    This is the 'microfiche card header' generation component.
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        print("Initializing Semantic Chunker...")
        print("  Loading sentence transformer model...")
        start = time.time()
        
        # Use lightweight model for fast prototyping
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        
        print(f"  ✓ Ready in {time.time() - start:.2f}s")
        
    def chunk_text(self, text: str, verbose: bool = True) -> List[Dict]:
        """
        Convert text into semantic supertokens.
        
        Args:
            text: Input text to chunk
            verbose: Print progress information
            
        Returns:
            List of supertoken dictionaries with metadata
        """
        if verbose:
            print(f"\nProcessing text ({len(text)} chars)...")
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if verbose:
            print(f"  Split into {len(sentences)} sentences")
        
        # Encode all sentences at once (efficient batch processing)
        if verbose:
            print("  Encoding sentences...")
        embeddings = self.encoder.encode(
            sentences, 
            show_progress_bar=verbose,
            convert_to_numpy=True
        )
        
        # Group by semantic similarity
        if verbose:
            print("  Grouping semantically...")
        chunks = self._group_semantically(sentences, embeddings)
        
        if verbose:
            print(f"  ✓ Created {len(chunks)} supertokens")
            self._print_stats(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        # Clean and filter empty
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _group_semantically(self, sentences: List[str], 
                          embeddings: np.ndarray) -> List[Dict]:
        """
        Group sentences into semantic chunks based on embedding similarity.
        This is the core 'microfiche card' creation logic.
        """
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for sentence, embedding in zip(sentences, embeddings):
            if current_embedding is None:
                # Start first chunk
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                # Calculate cosine similarity
                similarity = np.dot(current_embedding, embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(embedding)
                )
                
                # Decide: add to current chunk or start new one
                if similarity > self.similarity_threshold and len(current_chunk) < 10:
                    # High similarity - add to current chunk
                    current_chunk.append(sentence)
                    # Update running average of chunk embedding
                    current_embedding = (current_embedding + embedding) / 2
                else:
                    # Low similarity or chunk full - save and start new
                    if current_chunk:
                        chunks.append(
                            self._create_supertoken(current_chunk, current_embedding)
                        )
                    current_chunk = [sentence]
                    current_embedding = embedding
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_supertoken(current_chunk, current_embedding))
        
        return chunks
    
    def _create_supertoken(self, sentences: List[str], 
                          embedding: np.ndarray) -> Dict:
        """
        Create a supertoken structure - the 'microfiche card header'.
        
        Returns a dictionary with:
        - id: Unique identifier
        - summary: Short preview (what's on the card)
        - embedding: Semantic vector
        - confidence: How semantically coherent this chunk is
        - full_text: The actual content (stored separately in real system)
        - metadata: Various statistics
        """
        full_text = ' '.join(sentences)
        
        # Generate unique ID from content hash
        chunk_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
        
        # Calculate confidence based on intra-chunk coherence
        sentence_embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        variance = np.var([
            np.linalg.norm(e - embedding) 
            for e in sentence_embeddings
        ])
        confidence = 1.0 / (1.0 + variance)  # Higher = more coherent
        
        # Create summary (first sentence preview)
        summary = sentences[0]
        if len(summary) > 100:
            summary = summary[:97] + '...'
        
        return {
            'id': f'ST_{chunk_hash}',
            'summary': summary,
            'embedding': embedding,
            'confidence': float(confidence),
            'full_text': full_text,
            'metadata': {
                'sentence_count': len(sentences),
                'original_length': len(full_text),
                'compressed_length': len(summary),  # Simplified compression
                'compression_ratio': 1.0 - (len(summary) / len(full_text))
            }
        }
    
    def _print_stats(self, chunks: List[Dict]):
        """Print summary statistics"""
        total_original = sum(c['metadata']['original_length'] for c in chunks)
        total_compressed = sum(c['metadata']['compressed_length'] for c in chunks)
        avg_confidence = np.mean([c['confidence'] for c in chunks])
        
        print(f"\n  Statistics:")
        print(f"    Total original: {total_original} chars")
        print(f"    Total compressed: {total_compressed} chars")
        print(f"    Overall compression: {(1 - total_compressed/total_original)*100:.1f}%")
        print(f"    Average confidence: {avg_confidence:.3f}")


def demo():
    """Quick demonstration of the semantic chunker"""
    
    # Sample text about the microfiche concept itself
    sample_text = """
    Machine learning models face significant challenges with context length scaling 
    due to quadratic attention mechanisms. The computational cost grows exponentially 
    with sequence length, limiting practical applications.
    
    In order to address this limitation, researchers have proposed hierarchical 
    tokenization approaches. These methods compress semantic chunks into supertokens 
    while maintaining detailed information in separate storage tiers.
    
    The performance benefits are substantial when processing long documents. Models 
    can maintain much larger effective context windows without proportional increases 
    in memory usage or computation time.
    
    However, the implementation requires careful consideration of expansion policies 
    to avoid computational overhead. A controller must intelligently decide when to 
    retrieve full details versus operating on compressed representations.
    
    The future of AI depends on solving these efficiency challenges. Hierarchical 
    architectures like the microfiche array concept offer a promising path forward 
    for scalable language models.
    """
    
    print("=" * 70)
    print("MICROFICHE ARRAY PROTOTYPE - Semantic Chunker Demo")
    print("=" * 70)
    
    # Initialize chunker
    chunker = SemanticChunker(similarity_threshold=0.75)
    
    # Process text
    chunks = chunker.chunk_text(sample_text)
    
    # Display results
    print("\n" + "=" * 70)
    print("GENERATED SUPERTOKENS (Microfiche Cards):")
    print("=" * 70)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📇 Supertoken {i}: {chunk['id']}")
        print(f"   Confidence: {chunk['confidence']:.3f}")
        print(f"   Sentences: {chunk['metadata']['sentence_count']}")
        print(f"   Compression: {chunk['metadata']['original_length']} → "
              f"{chunk['metadata']['compressed_length']} chars "
              f"({chunk['metadata']['compression_ratio']*100:.1f}%)")
        print(f"   Summary: {chunk['summary']}")
        
        # Show if this would trigger expansion (low confidence)
        if chunk['confidence'] < 0.7:
            print(f"   ⚠️  LOW CONFIDENCE - would trigger expansion in real system")
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
