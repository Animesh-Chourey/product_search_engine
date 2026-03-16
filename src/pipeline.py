import os
import numpy as np
import torch
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel

class ProductSearchPipeline:
    def __init__(self, project_dir : str):
        self.project_dir = project_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = None
        self.processor = None
        
        self.text_index = None
        self.image_index = None
    

    def load_models(self):
        """
        Load CLIP model and processor
        This should be done ONCE per session
        """
        
        if self.model is None or self.processor is None:
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
    
    
    # -----------------------
    # Embedding utilities
    # -----------------------

    @staticmethod
    def _normalize(embedding : torch.Tensor) -> np.ndarray:
        """
            L2-normalize the embeddings for cosine similarity

            This ensures that dot product = cosine similarity,
            which is required for FAISS inner-product search.
        """
        
        # If embedding is a HuggingFace model output, extract tensor
        if hasattr(embedding, "pooler_output"):
            embedding = embedding.pooler_output

        # Ensure the embedding is a tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)

        # Normalize the vectors
        embedding = embedding / embedding.norm(dim = -1, keepdim = True)

        # Convert to numpy float32 for FAISS
        return embedding.detach().cpu().numpy()
    
    
    def embed_text(self, text : str) -> np.ndarray:
        """
        Generate CLIP embeddings for a single text input query
        """

        self.load_models()
        
        # Prepare input for CLIP
        inputs = self.processor(
            text = [text],
            return_tensors = "pt",
            padding = True,
            truncation = True
        ).to(self.device)

        with torch.no_grad():
            # Use CLIP feature extractor
            text_features = self.model.get_text_features(**inputs)

        return self._normalize(text_features)[0]
    

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate CLIP embedding for a single image
        """

        self.load_models()

        inputs = self.processor(
            images = image,
            return_tensors = "pt"
        ).to(self.device)

        with torch.no_grad():
            # Use CLIP feature extractor
            image_features = self.model.get_image_features(**inputs)

        return self._normalize(image_features)[0]
    

    # -----------------------
    # Batch Helpers(used in notebooks)
    # -----------------------

    def embed_text_batch(self, texts, batch_size=32):
        '''
        Embed a list of texts in batches
        Used during offline embedding generation 
        '''

        self.load_models()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i+batch_size]
            inputs = self.processor(
                text = batch,
                return_tensors = "pt",
                padding = True,
                truncation = True,
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            all_embeddings.append(self._normalize(text_features))

        return np.vstack(all_embeddings)
    
    def embed_image_batch(self, image_paths, batch_size = 32):
        """
        Embed a list of images in batches
        """

        self.load_models()
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i+batch_size]

            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self.processor(
                images = images,
                return_tensors = "pt",
            ).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            all_embeddings.append(self._normalize(image_features))

        return np.vstack(all_embeddings)
    

    def load_indexes(self):
        """
            Load FAISS indexes from disk

            This allows the search system to quickly retrieve 
            nearest product vectors without recomputing embeddings every time
        """

        if self.text_index is None or self.image_index is None:
            text_index_path = os.path.join(
                self.project_dir, "indexes/text_index.faiss"
            )
        
            image_index_path = os.path.join(
                self.project_dir, "indexes/image_index.faiss"
            )

            # Load FAISS indexes
            self.text_index = faiss.read_index(text_index_path)
            self.image_index = faiss.read_index(image_index_path)

    def search_text(self, query_text, top_k = 10):
        """
            Search products using a text query

            Steps:
                1. Convert the query to CLIP embeddings
                2. Search FAISS index
                3. Return top matching product indices
        """

        self.load_models()
        self.load_indexes()

        # Convert the query into embeddings
        # Reshape it into 2-d array so that FIASS library can perform similarity search
        query_embeddings = self.embed_text(query_text).reshape(1, -1)

        # Search FAISS index
        # Get "similarity scores" between query embeddings and top_k most similar text embeddings found in the index
        # Get "indices" corresponding to the top_k similar image embeddings.
        # We can use these indices to retrieve the most similar texts
        scores, indices = self.text_index.search(query_embeddings, top_k)

        return indices[0], scores[0]

    
    def search_image(self, image, top_k = 10):
        """
            Search products using an input image

            Steps:
                1. Convert input image into embeddings
                2. Query FAISS image index
                3. Return top matching products
        """

        self.load_models()
        self.load_indexes()

        # Convert the input image into vector representation (embeddings)
        # Reshape it into 2-d array so that FIASS library can perform similarity search
        query_image_embeddings = self.embed_image(image).reshape(1, -1)

        # Get "similarity scores" between query embeddings and top_k most similar image embeddings found in the index
        # Get "indices" corresponding to the top_k similar image embeddings.
        # We can use these indices to retrieve the most similar images or their metadata
        scores, indices = self.image_index.search(query_image_embeddings, top_k)

        return indices[0], scores[0]


    def search_multimodal(self, text = None, image = None, top_k = 10, alpha = 0.5):
        """
            Multimodal search combining text and image queries

            alpha = weight for the text
            1 - alpha = weight for image
        """

        self.load_models()
        self.load_indexes()

        final_scores = {}

        # if text query is provided
        # find similar items related to text provided
        # Took larger pool (top_k*2) of candidates to ensure not missing items scoring high in one modality
        # but not make the cut off  in the initial individual search    
        if text is not None:
            text_indices, text_scores = self.search_text(text, top_k*2)

            for idx, score in zip(text_indices, text_scores):
                final_scores[idx] = final_scores.get(idx, 0) + (alpha * score)
        
        # if image query is provided
        # finds similar items related to image provided
        # Took larger pool (top_k*2) of candidates to ensure not missing items scoring high in one modality
        # but not make the cut off  in the initial individual search
        if image is not None:
            image_indices, image_scores = self.search_image(image, top_k * 2)

            for idx, score in zip(image_indices, image_scores):
                final_scores[idx] = final_scores.get(idx, 0) + ((1-alpha) * score)

        # sort the tuples based on the 2nd element of each tuple (i.e. here "scores") 
        ranked_results = sorted(
            final_scores.items(),
            key = lambda x : x[1],
            reverse = True
        )

        return ranked_results[: top_k]