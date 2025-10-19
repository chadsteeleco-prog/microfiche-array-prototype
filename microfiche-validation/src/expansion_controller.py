class ExpansionController:
    """Intelligent expansion decision maker"""
    
    def __init__(self):
        self.uncertainty_threshold = 0.75
        self.attention_budget = 0.15
        self.expansion_history = []
        
    def should_expand(self, supertoken, context):
        """
        Decision factors:
        1. Confidence score (already have)
        2. Simulated attention weights
        3. Task type (code/math/general)
        4. Token position in sequence
        5. Expansion history (avoid thrashing)
        """
        
        # Uncertainty trigger
        if supertoken['confidence'] < self.uncertainty_threshold:
            return True, 'low_confidence'
            
        # Simulate attention weights
        attention = self.simulate_attention(supertoken, context)
        if attention > self.attention_budget:
            return True, 'high_attention'
            
        # Task-specific heuristics
        if self.is_code_or_math(supertoken):
            return True, 'task_specific'
            
        return False, 'keep_compressed'

