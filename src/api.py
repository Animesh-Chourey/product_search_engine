from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from PIL import Image
import io
import sys

# Add pipeline path
sys.path.append(".")

from src.pipeline import ProductSearchPipeline

# Initialize FastApi app
app = FastAPI(
    title = "Multimodal Product Search API",
    description = "Search products using text, image, or both",
)

# Initialize pipeline
pipeline = ProductSearchPipeline(project_dir=".")


# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
def root():
    return {"message" : "API is running"}


# ---------------------------
# Text Search Endpoint
# ---------------------------
@app.post("/search/text")
def search_text(query : str):
    """
        Search products using text query
    """

    results = pipeline.search_text(query, top_k = 5)

    # Load metadata
    pipeline.load_metadata()

    formatted_results = []

    for idx, score in results:
        idx = int(idx)
        score = float(score)

        product = pipeline.metadata.iloc[idx]

        # Build structured response
        formatted_results.append({
            "idx" : idx,
            "score" : score,
            "name" : product["productDisplayName"],
            "category" : product["articleType"],
            "image_path" : f"images/{idx}.jpg"
        })

    
    return {"results" : formatted_results}


# ---------------------------
# Image Search Endpoint
# ---------------------------
@app.post("/search/image")
async def search_image(file : UploadFile = File(...)):
    """
        Search Product using uploaded image
    """

    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    indices, scores = pipeline.search_image(image, top_k=5)

    return { "results": list(zip(indices.tolist(), scores.tolist())) }


# ---------------------------
# Multimodal Search Endpoint
# ---------------------------
@app.post("/search/multimodal")
async def search_multimodal(
    query: str = Form(None),
    file: UploadFile = File(None)
):
    """
        Search using both text + image
    """

    image = None

    if file is not None:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = pipeline.search_multimodal(
        text=query,
        image=image,
        top_k=5
    )


    formatted_results = [
    (int(idx), float(score)) for idx, score in results
    ]

    return {"results": formatted_results}