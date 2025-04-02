import json
import os
from pathlib import Path

import numpy as np

from src.types.face import Identity


class IdentityManager:
    def __init__(self, storage_path: str = "identities.json"):
        self.storage_path = storage_path
        self.identities: dict[str, Identity] = {}
        self.threshold = 0.6  # Similarity threshold for face matching
        self.load_identities()

    def load_identities(self):
        """Load identities from storage file."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path) as f:
                data = json.load(f)
                for name, identity_data in data.items():
                    embeddings = [np.array(emb) for emb in identity_data["embeddings"]]
                    self.identities[name] = Identity(
                        name=name, embeddings=embeddings, metadata=identity_data.get("metadata")
                    )

    def save_identities(self):
        """Save identities to storage file."""
        data = {}
        for name, identity in self.identities.items():
            data[name] = {"embeddings": [emb.tolist() for emb in identity.embeddings], "metadata": identity.metadata}
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def add_identity(self, name: str, embedding: np.ndarray, metadata: dict | None = None):
        """Add a new identity or update an existing one with a new embedding."""
        if name in self.identities:
            self.identities[name].embeddings.append(embedding)
            if metadata:
                self.identities[name].metadata = metadata
        else:
            self.identities[name] = Identity(name=name, embeddings=[embedding], metadata=metadata)
        self.save_identities()

    def remove_identity(self, name: str):
        """Remove an identity from the repository."""
        if name in self.identities:
            del self.identities[name]
            self.save_identities()

    def get_identity(self, name: str) -> Identity | None:
        """Get an identity by name."""
        return self.identities.get(name)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def find_best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Find the best matching identity for a given embedding."""
        best_name = None
        best_similarity = -1

        for name, identity in self.identities.items():
            # Compare with all embeddings of the identity
            for stored_embedding in identity.embeddings:
                similarity = self.compute_similarity(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name

        if best_similarity >= self.threshold:
            return best_name, best_similarity
        return None, best_similarity

    def add_identity_from_image(self, name: str, image_path: str, model_manager) -> bool:
        """Add an identity from an image file."""
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            return False

        # Get face embeddings
        faces = model_manager.models["insightface"].app.get(image)
        if not faces:
            return False

        # Use the first face detected
        embedding = faces[0].embedding
        self.add_identity(name, embedding)
        return True

    def add_identities_from_directory(self, directory: str, model_manager) -> dict[str, bool]:
        """Add identities from a directory of images."""
        results = {}
        for image_path in Path(directory).glob("*.jpg"):
            name = image_path.stem
            success = self.add_identity_from_image(name, str(image_path), model_manager)
            results[name] = success
        return results
