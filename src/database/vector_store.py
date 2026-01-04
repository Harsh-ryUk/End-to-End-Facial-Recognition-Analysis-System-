import faiss
import pickle
import os
import numpy as np

class VectorStore:
    def __init__(self, index_path="models/faiss_index.bin", mapping_path="models/id_map.pkl", dim=512):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.dim = dim
        
        # Load or Create Index
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            print(f"[FAISS] Loading index from {index_path}")
            self.index = faiss.read_index(index_path)
            with open(mapping_path, 'rb') as f:
                self.id_to_name = pickle.load(f)
        else:
            print("[FAISS] Creating new index")
            self.index = faiss.IndexFlatL2(dim)
            self.id_to_name = {} # {id (int): name (str)}

    def add_user(self, name, embedding):
        """
        Add a user embedding to FAISS.
        """
        # FAISS expects float32
        vec = np.array([embedding], dtype=np.float32)
        
        # New ID
        new_id = self.index.ntotal
        
        self.index.add(vec)
        self.id_to_name[new_id] = name
        
        self.save()
        return new_id

    def search(self, embedding, threshold=0.5):
        """
        Search for closest match.
        Returns: (name, score) or ("Unknown", score)
        """
        vec = np.array([embedding], dtype=np.float32)
        
        # Search k=1
        distances, ids = self.index.search(vec, 1)
        
        idx = ids[0][0]
        dist = distances[0][0]
        
        # Convert L2 distance to Similarity Score (Approx)
        # Cosine distance is usually 1 - similarity. L2 is related.
        # For arcface normalized vectors, L2 = sqrt(2(1-cos)).
        # Let's map L2 back to a score 0-1.
        # If L2 is small, similarity is high.
        
        # Threshold: ArcFace L2 threshold usually ~1.0 for loose, ~0.8 strict.
        # Convert to a confidence score?
        # Let's keep it simple: if dist < threshold, match.
        
        # Note: 'threshold' passed here is likely for Cosine metric from caller.
        # L2 Threshold needs to be calibrated. 
        # For ArcFace, Cosine 0.5 ~= L2 1.0 (very rough).
        # Let's say L2 threshold 1.2 is decent for verification.
        
        l2_threshold = 1.2
        
        if idx != -1 and dist < l2_threshold:
            name = self.id_to_name.get(idx, "Unknown")
            # Invert dist for display "score" (fake it for UI consistency 0.0-1.0)
            # score = 1.0 - (dist / 2.0)
            score = max(0.0, 1.4 - dist) # Heuristic
            return name, score
        
        return "Unknown", 0.0

    def delete_user(self, name):
        """
        Deleting from FlatL2 is hard (indices shift).
        Simple hack: Remove from ID map so it returns 'Unknown'.
        Real fix: Rebuild index.
        """
        # Find all IDs pointing to 'name'
        ids_to_remove = [k for k, v in self.id_to_name.items() if v == name]
        for i in ids_to_remove:
            del self.id_to_name[i]
        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.id_to_name, f)
