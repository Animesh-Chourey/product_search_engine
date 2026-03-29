# Product Search Engine
End-to-end **multimodal product search engine** that allows the user to search products using test, images, or both, built with **CLIP (Computer Vision + NLP)**, **FAISS**, **FastAPI + Streamlit**.  This repository documents the complete ML lifecycle, starting with data ingestion, preprocessing, and exploratory analysis on Google Colab


## Project Overview
This project aims to build an end-to-end multimodal product search engine that allows users to search for product using text queries, images, or both. The system leverages **Computer Vision** and **Natural Language Processing** using transformer based model **CLIP**(Contrastive Language-Image Pretraining) to map texts and images into a shared embedding space, enabling efficient similarity-based retrieval.

The project is designed to closely mimic e-commerce search systems, used by companies such as Amazon, Zalando, and Flipkart, with a strong focus on production ready ML practices i.e. modular pipelines, scalable embeddings, & API-based development


### Dataset
This project uses the **Fashion Product Images (Small)** dataset sourced from Kaggle.
#### Dataset Description
The dataset contains 44,000+ fashion product images along with structured metadata commonly found in e-commerce catalogs. Each product is represented by:
- A high-quality product image
- Product name and category information
- Attributes such as gender, season, color, and usage

Link: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)


### Step - 1 : Embedding Pipeline
- Implemented reusable CLIP-based embedding pipeline
- Centralized image and text embedding logic
- Added batch processing for scalable offline embedding
- Ensured cosine-normalized embeddings for retrieval


### Step - 2 : Vector Search
Built FAISS vector indexes to enable fast nearest-neighbor search over CLIP embeddings.

Indexes created:
- text_index.faiss
- image_index.faiss

These indexes allow millisecond-level similarity search.


### Step - 3 : Multimodal Search Engine
Implemented full search pipeline:
- Text search
- Image search
- Multimodal fusion search

Search uses FAISS vector indexes built from CLIP embeddings.


### Step - 4 : Improvements
- Added hybrid retrieval (CLIP similarity + keyword matching)
- Implemented score normalization for multimodal fusion so that neither text query nor image query dominates randomly 
- Improved query understanding with token-based matching
- Added caching for faster repeated queries


### Step - 5 : Deployment & UI

#### FastAPI Backend
Handles all inference requests
- Supports:
  - Text search
  - Image search
  - Multimodal search
- Loads FAISS indexes and CLIP model for the real time retrieval

#### Streamlit Frontend
- Interactive UI for users
- Supports:
  - Text queries
  - Image uploads
  - Combined multimodal search
- Communicates with FastAPI via REST API calls


### How to Run the Full System

#### Start Backend API

```bash
uvicorn src.api:app --reload
```

Runs at : http://127.0.0.1:8000

#### Start Frontend UI
```bash
streamlit run src/ui.py
```

Runs at : streamlit run src/streamlit_ui.py