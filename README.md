# Product Search Engine
End-to-end multimodal product search engine using Computer Vision and NLP, built with CLIP, FastAPI, &amp; Gradio.  This repository documents the complete ML lifecycle, starting with data ingestion, preprocessing, and exploratory analysis on Google Colab


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