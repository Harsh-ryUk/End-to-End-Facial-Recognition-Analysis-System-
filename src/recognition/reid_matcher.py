import numpy as np
import pickle
import os
from ..database.vector_store import VectorStore

class ReidMatcher:
    """
    Face Re-Identification Matcher.
    Uses Cosine Similarity to match embeddings against a registered database.
    """
    def __init__(self, db_path='models/face_db.pkl'):
        # db_path arg kept for compatibility, but we rely on VectorStore paths
        self.db = VectorStore()
        
    def register(self, name, embedding):
        """
        Add new identity.
        """
        self.db.add_user(name, embedding)
        return True
        
    def delete(self, name):
        """
        Delete identity.
        """
        self.db.delete_user(name)
        return True

    def match(self, embedding):
        """
            tuple: (name, score) or (None, score) if below threshold.
        """
        if not self.database:
            return "Unknown", 0.0
            
        best_score = -1.0
        best_name = "Unknown"
        
        for name, db_emb in self.database.items():
            # Cosine similarity: dot product (since embeddings are normalized)
            score = np.dot(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name = name
                
        if best_score > self.threshold:
            return best_name, best_score
        else:
            return "Unknown", best_score
