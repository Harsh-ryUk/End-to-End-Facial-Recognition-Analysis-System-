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
        Compare embedding against FAISS index.
        Returns: (name, score)
        """
        if embedding is None:
            return "Unknown", 0.0
            
        name, score = self.db.search(embedding)
        return name, score
