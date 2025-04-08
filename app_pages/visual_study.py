import streamlit as st
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import numpy as np
import random
import itertools
import io
from PIL import Image as PILImage

def app():
    st.title("Visual Differentiation Study")
    st.markdown("""
    This section highlights the visual characteristics of healthy and mildew-infected cherry leaves using image processing techniques.
    """)

    # Average & variability images
    if st.checkbox("Show Average and Variability Images"):
        st.image(
            "outputs/02_data_visualisation/avg_var_healthy.png",
            caption="Healthy Leaf: Average & Variability",
        width=500)
        st.image(
            "outputs/02_data_visualisation/avg_var_powdery_mildew.png",
            caption="Powdery Mildew Leaf: Average & Variability",
        width=500)
        st.info("There is a clear visual difference in texture and brightness between healthy and infected leaves.")

    # Difference between average images
    if st.checkbox("Show Difference Between Average Images"):
        st.image(
            "outputs/02_data_visualisation/avg_diff.png",
            caption="Difference Between Average Images",
        width=1000)
        st.warning("Although subtle, the darker and green-centered regions help differentiate mildew infection.")

        # Image Montage Section
    if st.checkbox("Generate Image Montage"):
        label_options = ["healthy", "powdery_mildew"]
        selected_label = st.selectbox("Select Leaf Condition", label_options)
        nrows = st.slider("Number of rows", 1, 6, 4)
        ncols = st.slider("Number of columns", 1, 6, 3)
        figsize_val = st.slider("Figure size (width, height)", 5, 20, (12, 10))

        if st.button("Create Montage"):
            dir_path = os.path.join("inputs/cherry_leaves_split/train", selected_label)

            if os.path.exists(dir_path):
                montage_buffer = image_montage(
                    dir_path="inputs/cherry_leaves_split/train",
                    label=selected_label,
                    nrows=nrows, ncols=ncols,
                    figsize=figsize_val
                )
                if montage_buffer:
                    st.download_button(
                        label="Download Montage as PNG",
                        data=montage_buffer,
                        file_name="montage.png",
                        mime="image/png"
                    )
                else:
                    st.warning("No montage generated — maybe no images found.")
            else:
                st.error(f"Directory not found for the selected label: `{dir_path}`.\nThis feature is unavailable in the deployed version.")


    # Image dimension distribution
    if st.checkbox("Show Image Dimension Distribution"):
        label_dirs = ["healthy", "powdery_mildew"]
        all_widths, all_heights = [], []

        dataset_available = True
        for lbl in label_dirs:
            path = os.path.join("inputs/cherry_leaves_split/train", lbl)
            if os.path.exists(path):
                for file in os.listdir(path):
                    img = imread(os.path.join(path, file))
                    all_heights.append(img.shape[0])
                    all_widths.append(img.shape[1])
            else:
                dataset_available = False
                st.warning(f"Dataset folder not found: `{path}`. Skipping this class.")

        if dataset_available and all_widths:
            fig, ax = plt.subplots(figsize=(2, 1))
            sns.scatterplot(x=all_widths, y=all_heights, alpha=0.5, ax=ax)
            ax.set_title("Image Dimension Distribution")
            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Height (px)")
            st.pyplot(fig)
            st.info(f"Total images analyzed: {len(all_widths)}")
        else:
            st.warning("    No image dimensions could be plotted. Dataset folders are missing in this environment.")


    # Image intensity histogram
    if st.checkbox("Show Image Intensity Histogram"):
        label = st.selectbox("Select Label for Histogram", ["healthy", "powdery_mildew"], key="hist")
        path = os.path.join("inputs/cherry_leaves_split/train", label)

        if os.path.exists(path):
            pixel_values = []
            try:
                for file in os.listdir(path)[:50]:
                    img = imread(os.path.join(path, file))
                    gray = np.mean(img, axis=2)
                    pixel_values.extend(gray.flatten())

                fig, ax = plt.subplots()
                sns.histplot(pixel_values, bins=50, kde=True, ax=ax, color="gray")
                ax.set_title(f"Pixel Intensity Distribution - {label.capitalize()}")
                ax.set_xlabel("Pixel Intensity")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred while processing images: {str(e)}")

        else:
            st.warning(f"Dataset folder not found: `{path}`.\nThis feature is unavailable in the deployed app.")


def image_montage(dir_path, label, nrows, ncols, figsize=(12, 10)):
    sns_dir = os.path.join(dir_path, label)
    if not os.path.exists(sns_dir):
        st.error("Directory not found for the selected label.")
        return None

    images = os.listdir(sns_dir)
    if nrows * ncols > len(images):
        st.warning(f"Not enough images in the folder. Found {len(images)}.")
        return None

    selected_images = random.sample(images, nrows * ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plot_positions = list(itertools.product(range(nrows), range(ncols)))

    for i, pos in enumerate(plot_positions):
        img_path = os.path.join(sns_dir, selected_images[i])
        img = imread(img_path)
        axes[pos[0], pos[1]].imshow(img)
        axes[pos[0], pos[1]].axis("off")
        axes[pos[0], pos[1]].set_title(f"{img.shape[1]}x{img.shape[0]} px")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.pyplot(fig)
    return buf