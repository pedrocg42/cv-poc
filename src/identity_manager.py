import json
import os
from pathlib import Path

import numpy as np

from src.types.face import Identity
from src.types.selfie_data import SelfieData


class IdentityManager:
    def __init__(self, storage_path: str = ".identities/identities.json"):
        self.storage_path = storage_path
        self.identities: dict[int, Identity] = {}  # Changed to use ID as key
        self.threshold = 0.3  # Similarity threshold for face matching
        self.load_identities()

    def load_identities(self):
        """Load identities from storage file and their embeddings from folders."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path) as f:
                identities_data = json.load(f)
                for identity_data in identities_data:
                    # Create identity folder path
                    identity_folder = Path(".identities") / str(identity_data["id"])

                    # Load embeddings from the identity folder
                    embeddings = []
                    if identity_folder.exists():
                        embeddings = np.load(identity_folder / "embeddings.npy")

                        # Create Identity object
                        self.identities[identity_data["id"]] = Identity(
                            id=identity_data["id"],
                            name=identity_data["name"],
                            last_name=identity_data["last_name"],
                            embedding=embeddings,
                        )

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        similarity = np.dot(embedding1, embedding2.T)
        return similarity

    def find_best_match(self, embedding: np.ndarray) -> tuple[int | None, float]:
        """Find the best matching identity for a given embedding."""
        best_id = None
        best_similarity = -1

        for identity_id, identity in self.identities.items():
            # Compare with all embeddings of the identity
            for stored_embedding in identity.embedding:
                similarity = self.compute_similarity(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = identity_id

        return best_id, best_similarity

    def identify(self, selfie_data: SelfieData) -> SelfieData:
        if not selfie_data.faces:
            return selfie_data

        for face in selfie_data.faces:
            if not face.identity:
                continue

            best_id, best_similarity = self.find_best_match(face.identity.embedding)

            if best_similarity > self.threshold:
                new_identity = self.identities[best_id].model_copy(deep=True)
                new_identity.embedding = face.identity.embedding
                face.identity = new_identity

        return selfie_data
