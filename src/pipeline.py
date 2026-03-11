import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ProductSearchPipeline:
    def __init__(self, project_dir : str):
        self.project_dir = project_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = None
        self.processor = None
        

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
                "openai-clip-vit-base-patch32"
            )
    
    
    # -----------------------
    # Embedding utilities
    # -----------------------

    @staticmethod
    def _normalize(embedding : torch.Tensor) -> np.ndarray:
        """
        L2-normalize the embeddings for cosine similarity
        """
        embedding = embedding / embedding.norm(dim = -1, keepdim = True)
        return embedding.cpu().numpy()
    
    
    def embed_text(self, text : str) -> np.ndarray:
        """
        Generate CLIP embeddings for a single text input 
        """

        self.load_models()
        
        inputs = self.processor(
            text = [text],
            return_tensors = "pt",
            padding = True,
            truncation = True
        ).to(self.device)

        with torch.no_grad():
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