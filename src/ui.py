import streamlit as st
import requests
from PIL import Image
import io
import os

# Base URl for the backend API
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Product Search", layout="wide")

st.title("Multimodal Product Search Engine")

# -------------------------
# User input section
# -------------------------

col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Enter search query")

with col2:
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

    with st.spinner("Searching..."):

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

            if len(results) == 0:
                st.warning("No results found")

            # Display results in grid layout
            cols = st.columns(5)  # 5 products per row

            for i, item in enumerate(results):
                with cols[i % 5]:
                    try:
                        # Extract data from API response
                        product_id = item["idx"]
                        score = item["score"]
                        name = item["name"]
                        category = item["category"]
                        image_path = os.path.join(os.getcwd(), "images", f"{product_id}.jpg")


                        # Display image if exists
                        if os.path.exists(image_path):
                            st.image(image_path, width=150)
                        else:
                            st.write("No Image")

                        # Display product info
                        st.markdown(f"**{name}**")
                        st.caption(f"{category}")
                        st.write(f"Score: {score}")

                    except Exception as e:
                        st.error(f"Error displaying item: {e}")
                        st.write(item)

        except Exception as e:
            st.error(f"Error: {str(e)}")