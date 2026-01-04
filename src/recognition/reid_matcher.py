import numpy as np
import pickle
import os

class ReidMatcher:
    """
    Face Re-Identification Matcher.
    Uses Cosine Similarity to match embeddings against a registered database.
    """
    def __init__(self, db_path="models/face_db.pkl", threshold=0.4):
        self.db_path = db_path
        self.threshold = threshold
        self.database = {} # {name: embedding}
        self.load_db()
        
    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.database = pickle.load(f)
                print(f"Loaded {len(self.database)} identities from {self.db_path}")
            except Exception as e:
                print(f"Error loading DB: {e}")
                self.database = {}
                
    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
            
    def register(self, name, embedding):
        """Register a new identity if not already present."""
        # Check if already registered
        existing_name, score = self.match(embedding)
        if existing_name != "Unknown":
            print(f"Face already registered as {existing_name} (Score: {score:.2f}). Skipping.")
            return existing_name
            
        self.database[name] = embedding
        self.save_db()
        print(f"Registered {name}")
        return name
        
    def delete(self, name):
        """Delete an identity from the database."""
        if name in self.database:
            del self.database[name]
            self.save_db()
            print(f"Deleted {name}")
            return True
        else:
            print(f"Identity {name} not found.")
            return False

    def match(self, embedding):
        """
        Find best match in database.
        
        Returns:
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
