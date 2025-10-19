#!/usr/bin/env python3
"""
Microfiche Array Prototype - Tiered Storage Manager
Day 1 Component: Simulate GPU/RAM/Storage hierarchy
"""

import json
import os
from typing import Dict, Optional, List
import time
from pathlib import Path

class TieredStorageManager:
    """
    Simulates the 3-tier memory hierarchy:
    - Tier 1: GPU/HBM (fast, limited) - stores supertoken headers
    - Tier 2: System RAM (medium) - caches recently expanded details
    - Tier 3: SSD/NVMe (slow, large) - stores all full-text details
    
    This is the 'microfiche drawer' system.
    """
    
    def __init__(self, storage_dir: str = './storage'):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Tier 1: HBM/GPU memory simulation (supertoken headers only)
        self.tier1_cache = {}
        self.tier1_capacity = 20  # Limited capacity
        self.tier1_access_order = []  # For LRU eviction
        
        # Tier 2: RAM cache (expanded details)
        self.tier2_cache = {}
        self.tier2_capacity = 50  # Larger capacity
        self.tier2_access_order = []
        
        # Tier 3: Disk storage (all full details)
        # Stored as files in storage_dir
        
        # Performance metrics
        self.metrics = {
            'tier1_hits': 0,
            'tier2_hits': 0,
            'tier3_hits': 0,
            'total_expansions': 0,
            'storage_operations': 0,
            'tier1_evictions': 0,
            'tier2_evictions': 0
        }
        
        print(f"Storage Manager initialized:")
        print(f"  Tier 1 (HBM): {self.tier1_capacity} supertokens")
        print(f"  Tier 2 (RAM): {self.tier2_capacity} expanded chunks")
        print(f"  Tier 3 (SSD): {self.storage_dir}")
    
    def store_supertoken(self, supertoken: Dict) -> None:
        """
        Store a supertoken across the tiered hierarchy.
        Header goes to Tier 1, full details to Tier 3.
        """
        token_id = supertoken['id']
        self.metrics['storage_operations'] += 1
        
        # Create lightweight header for Tier 1 (HBM)
        header = {
            'id': token_id,
            'summary': supertoken['summary'],
            'confidence': supertoken['confidence'],
            'compressed_length': supertoken['metadata']['compressed_length']
        }
        
        # Store in Tier 1 with LRU eviction
        self._store_tier1(token_id, header)
        
        # Store full details in Tier 3 (disk)
        self._store_tier3(token_id, supertoken)
    
    def _store_tier1(self, token_id: str, header: Dict) -> None:
        """Store in Tier 1 (HBM) with LRU eviction"""
        # Check capacity
        if len(self.tier1_cache) >= self.tier1_capacity:
            # Evict least recently used
            if self.tier1_access_order:
                evict_id = self.tier1_access_order.pop(0)
                del self.tier1_cache[evict_id]
                self.metrics['tier1_evictions'] += 1
        
        self.tier1_cache[token_id] = header
        self.tier1_access_order.append(token_id)
    
    def _store_tier3(self, token_id: str, supertoken: Dict) -> None:
        """Store full details in Tier 3 (disk)"""
        # Remove numpy arrays for JSON serialization
        data_to_store = {
            'id': supertoken['id'],
            'summary': supertoken['summary'],
            'confidence': supertoken['confidence'],
            'full_text': supertoken['full_text'],
            'metadata': supertoken['metadata']
        }
        
        file_path = self.storage_dir / f"{token_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data_to_store, f, indent=2)
    
    def expand_supertoken(self, token_id: str) -> Optional[Dict]:
        """
        Retrieve full supertoken details (expansion operation).
        This is 'pulling the microfiche film from the drawer'.
        """
        self.metrics['total_expansions'] += 1
        
        # Check Tier 2 first (RAM cache)
        if token_id in self.tier2_cache:
            self.metrics['tier2_hits'] += 1
            self._update_tier2_access(token_id)
            return self.tier2_cache[token_id]
        
        # Check Tier 3 (disk)
        return self._expand_from_tier3(token_id)
    
    def _expand_from_tier3(self, token_id: str) -> Optional[Dict]:
        """Fetch from Tier 3 and promote to Tier 2"""
        self.metrics['tier3_hits'] += 1
        
        file_path = self.storage_dir / f"{token_id}.json"
        
        if not file_path.exists():
            return None
        
        # Simulate disk latency (1-5ms)
        time.sleep(0.002)
        
        # Load from disk
        with open(file_path, 'r') as f:
            detail = json.load(f)
        
        # Promote to Tier 2 (RAM cache)
        self._promote_to_tier2(token_id, detail)
        
        return detail
    
    def _promote_to_tier2(self, token_id: str, detail: Dict) -> None:
        """Promote expanded detail to Tier 2 cache"""
        # Check capacity
        if len(self.tier2_cache) >= self.tier2_capacity:
            # Evict least recently used
            if self.tier2_access_order:
                evict_id = self.tier2_access_order.pop(0)
                del self.tier2_cache[evict_id]
                self.metrics['tier2_evictions'] += 1
        
        self.tier2_cache[token_id] = detail
        self.tier2_access_order.append(token_id)
    
    def _update_tier2_access(self, token_id: str) -> None:
        """Update LRU order for Tier 2"""
        if token_id in self.tier2_access_order:
            self.tier2_access_order.remove(token_id)
        self.tier2_access_order.append(token_id)
    
    def get_header(self, token_id: str) -> Optional[Dict]:
        """Get supertoken header from Tier 1 (fast operation)"""
        if token_id in self.tier1_cache:
            self.metrics['tier1_hits'] += 1
            return self.tier1_cache[token_id]
        return None
    
    def get_all_headers(self) -> List[Dict]:
        """Get all supertoken headers from Tier 1"""
        return list(self.tier1_cache.values())
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        total_hits = (self.metrics['tier1_hits'] + 
                     self.metrics['tier2_hits'] + 
                     self.metrics['tier3_hits'])
        
        cache_hits = self.metrics['tier1_hits'] + self.metrics['tier2_hits']
        cache_hit_rate = (cache_hits / total_hits * 100) if total_hits > 0 else 0
        
        expansion_rate = (self.metrics['total_expansions'] / 
                         self.metrics['storage_operations'] * 100 
                         if self.metrics['storage_operations'] > 0 else 0)
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'expansion_rate': expansion_rate,
            'tier1_utilization': len(self.tier1_cache) / self.tier1_capacity * 100,
            'tier2_utilization': len(self.tier2_cache) / self.tier2_capacity * 100
        }
    
    def print_metrics(self) -> None:
        """Print formatted metrics"""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("STORAGE PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Total Operations: {metrics['storage_operations']}")
        print(f"Total Expansions: {metrics['total_expansions']}")
        print(f"Expansion Rate: {metrics['expansion_rate']:.1f}%")
        print(f"\nCache Performance:")
        print(f"  Tier 1 Hits (HBM): {metrics['tier1_hits']}")
        print(f"  Tier 2 Hits (RAM): {metrics['tier2_hits']}")
        print(f"  Tier 3 Hits (SSD): {metrics['tier3_hits']}")
        print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"\nMemory Utilization:")
        print(f"  Tier 1: {metrics['tier1_utilization']:.1f}%")
        print(f"  Tier 2: {metrics['tier2_utilization']:.1f}%")
        print(f"\nEvictions:")
        print(f"  Tier 1: {metrics['tier1_evictions']}")
        print(f"  Tier 2: {metrics['tier2_evictions']}")
        print("=" * 60)


def demo():
    """Demonstrate tiered storage system"""
    
    print("=" * 70)
    print("MICROFICHE ARRAY PROTOTYPE - Tiered Storage Demo")
    print("=" * 70)
    print()
    
    # Initialize storage
    storage = TieredStorageManager(storage_dir='./storage/demo')
    print()
    
    # Create mock supertokens
    print("Creating and storing 30 supertokens...")
    supertokens = []
    for i in range(30):
        token = {
            'id': f'ST_{i:04d}',
            'summary': f'Summary of semantic chunk {i}',
            'confidence': 0.7 + (i % 3) * 0.1,  # Varying confidence
            'full_text': f'Full text content of chunk {i}. ' * 20,
            'metadata': {
                'sentence_count': 3 + (i % 5),
                'original_length': 500 + i * 10,
                'compressed_length': 100 + i * 2
            }
        }
        supertokens.append(token)
        storage.store_supertoken(token)
    
    print(f"✓ Stored {len(supertokens)} supertokens")
    print(f"  (Tier 1 can only hold {storage.tier1_capacity}, older ones evicted)")
    print()
    
    # Simulate various access patterns
    print("Simulating access patterns...")
    print()
    
    # Pattern 1: Sequential access (cold cache)
    print("1. Sequential access (testing Tier 3 → Tier 2 promotion):")
    for i in [0, 1, 2]:
        detail = storage.expand_supertoken(f'ST_{i:04d}')
        print(f"   Expanded ST_{i:04d} - Cache: {'HIT' if i > 0 else 'MISS'}")
    print()
    
    # Pattern 2: Repeated access (hot cache)
    print("2. Repeated access (testing Tier 2 cache hits):")
    for _ in range(3):
        detail = storage.expand_supertoken('ST_0001')
    print(f"   Expanded ST_0001 3x - Should be 2 cache hits")
    print()
    
    # Pattern 3: Working set (realistic usage)
    print("3. Working set pattern (realistic mixed access):")
    import random
    working_set = [f'ST_{i:04d}' for i in [5, 6, 7, 8, 9]]
    for _ in range(10):
        token_id = random.choice(working_set)
        storage.expand_supertoken(token_id)
    print(f"   Expanded 10 tokens from working set of 5")
    print()
    
    # Print final metrics
    storage.print_metrics()
    
    # Show what's currently in each tier
    print("\nCurrent Tier Status:")
    print(f"  Tier 1 (Headers): {len(storage.tier1_cache)} items")
    print(f"  Tier 2 (Expanded): {len(storage.tier2_cache)} items")
    print(f"  Tier 3 (Disk): {len(list(storage.storage_dir.glob('*.json')))} files")
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
