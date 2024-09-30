import requests
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas, CanvasResult

from utils import markdown_progress

# main size of image
mnist_size = 28

# separate on canvas and probabilities columns, set wide layout
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)

# canvas component
with col1:
    canvas_result: CanvasResult = st_canvas(
        # fill_color="#eee",  # "rgba(255, 165, 0, 0.3)" fixed fill color with some opacity
        stroke_width=st.sidebar.slider("Stroke width: ", 10, 50, 30, 5),
        stroke_color=st.sidebar.color_picker("Stroke color hex: ", "#fff"),
        background_color=st.sidebar.color_picker(
            "Background color hex: ", "#000"
        ),  # "#eee"
        # background_image=Image.open(bg_image) if bg_image else None,
        # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        update_streamlit=st.sidebar.checkbox("Update in realtime", True),
        height=mnist_size * 20,
        width=mnist_size * 20,
        drawing_mode=st.sidebar.selectbox(
            "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
        ),
        # point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        key="canvas",
    )

    image: np.ndarray | None = canvas_result.image_data
    if image is not None and image[:, :, 0:3].any():  # exclude alpha channel with 255
        h, w, _ = image.shape
        assert h == w
        block_size = w // mnist_size

        # convert from numpy to PIL Image grayscale via ITU-R 601-2 luma transform
        im = Image.fromarray(image, mode="RGBA").convert("L")

        # max pull from image to resize it to 28 by 28
        im.thumbnail(size=(mnist_size, mnist_size))

        # optional part to save the image
        # st.image(im, width=mnist_size * 20)
        # im.save("image.png")

        # get back image as numpy of shape (28, 28)
        image = np.array(im)

        # get results from API
        results = requests.post("http://api:8000/predict", data=image.tobytes())
        response = results.json()

        probs = response.get("probs")
        pred = response.get("pred")
    else:
        # draw equil probs on empty image
        probs = [0.1 for _ in range(10)]


# prediction
with col2:
    df = pd.DataFrame(data={"class": range(10), "prob": probs})
    df["prob"] = df["prob"].round(2)
    df["confidence"] = df["prob"].map(markdown_progress)
    st.markdown(body="""""" + df.to_markdown(index=False), unsafe_allow_html=True)
