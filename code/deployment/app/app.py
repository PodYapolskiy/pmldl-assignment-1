import pandas as pd
import numpy as np
from PIL import Image
import skimage.measure
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import softmax, markdown_progress

# ...
mnist_size = 28

# separate on canvas and probabilities columns, set wide layout
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)

# canvas component
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=st.sidebar.slider("Stroke width: ", 10, 50, 25, 5),
        stroke_color=st.sidebar.color_picker("Stroke color hex: "),
        background_color=st.sidebar.color_picker("Background color hex: ", "#eee"),
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

    if canvas_result.image_data is not None:
        image: np.ndarray = canvas_result.image_data
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
        print(image.shape)

        # new_image = image.resize(image_height, image_height)
        if image.shape == (mnist_size, mnist_size):
            print(image.shape)


# prediction
with col2:
    probs = softmax(np.random.randn(10))
    df = pd.DataFrame(data={"class": range(10), "prob": probs})
    df["prob"] = df["prob"].round(2)
    df["confidence"] = df["prob"].map(markdown_progress)
    st.markdown(body="""""" + df.to_markdown(index=False), unsafe_allow_html=True)


# Do something interesting with the image data and paths
# if canvas_result.json_data is not None:
#     print(canvas_result.image_data.shape)
# objects = pd.json_normalize(
#     canvas_result.json_data["objects"]
# )  # need to convert obj to str because PyArrow
# for col in objects.select_dtypes(include=["object"]).columns:
#     objects[col] = objects[col].astype("str")
# st.dataframe(objects)
# st.dataframe(canvas_result.image_data)
