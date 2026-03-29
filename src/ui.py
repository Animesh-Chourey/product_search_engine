import streamlit as st
import requests
from PIL import Image
import io

# Base URl for the backend API
API_URL = "http://127.0.0.1:8000"

st.title("Multimodal Product Search Engine")

# -------------------------
# User input section
# -------------------------

query = st.text_input("Enter search query")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# If user uploads an image
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# -------------------------
# Search Button
# -------------------------
if st.button("Search"):

    try:
        files = None

        # Prepare Image for API if it exists
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            files = {"file": ("image.png", buffered.getvalue(), "image/png")}

        # Decide which endpoint to call

        # Multimodal Search API call
        if image is not None and query:
            response = requests.post(
                f"{API_URL}/search/multimodal",
                data={"query": query},
                files=files
            )
        
        # Image Search API call
        elif image is not None:
            response = requests.post(
                f"{API_URL}/search/image",
                files=files
            )

        # Text Search API call
        else:
            response = requests.post(
                f"{API_URL}/search/text",
                params={"query": query}
            )

        # -------------------------
        # Process API Response
        # -------------------------

        # COnvert response into JSON and get the "results"
        results = response.json().get("results", [])

        st.subheader("Results")

        for item in results:
            try:
                idx, score = item
                # Display formatted result
                st.write(f"Product {idx}, \t Score: {round(float(score), 3)}")
            except:
                st.write(item)

    except Exception as e:
        st.error(f"Error: {str(e)}")