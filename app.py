import io
import streamlit as st
import tensorflow as tf
from PIL import Image
from transformers import pipeline
from typing import Optional, Dict


def load_image() -> Optional[Image.Image]:
    """Upload and display an image.

    Returns:
        Optional[Image.Image]: The uploaded image or None if no image was uploaded.
    """
    uploaded_file = st.file_uploader('Upload image here')
    if uploaded_file is not None:
        st.write("Filename:", uploaded_file.name)
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    return None


def preprocess_image(img: Image.Image) -> tf.Tensor:
    """Preprocess the uploaded image for model prediction.

    Args:
        img (Image.Image): The image to preprocess.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """
    img = img.resize((100, 100))
    x = tf.keras.utils.img_to_array(img)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = tf.expand_dims(x, axis=0)  # Add batch dimension
    return x

def load_model():
    return pipeline(model="JuanMa360/room-classification")

# Project Title
st.title("Room Classification Project")

# Team Member
st.write("""
         House & Apartaments Classification model
         #### TEAM MEMBER
         - Сидоркин Георгий Владимирович РИМ-130908
         - Романова Виктория РИМ-130908
         - Гребнев Никита РИМ-130908
         - Шлёгин Лев Русланович РИМ-130908
         """)

st.write("""#### Our Project""")

# Initial function
loadedImage = load_image()
model = load_model()

result = st.button('Submit')

if result:
    x = preprocess_image(loadedImage)
    prediction = model.predict(loadedImage)
    st.write("""#### Output""")
    st.write(prediction)

def display_project_title(title: str):
    """Display the project title.

    Args:
        title (str): The title of the project.
    """
    st.title(title)


def display_team_info(team_info: str):
    """Display the information about the team.

    Args:
        team_info (str): The markdown string containing team information.
    """
    st.write(team_info)
def save_results(results: Dict[str, float], filename: str = "results.txt"):
    """Save the classification results to a text file.
    
    Args:
        results (Dict[str, float]): The classification results.
        filename (str): The name of the file to save the results.
    """
    with open(filename, "w") as file:
        for room_type, score in results.items():
            file.write(f"{room_type}: {score:.2f}\n")
    st.write(f"Results saved to {filename}")

def main():
    """Main function to run the Streamlit app."""
    display_project_title("Room Classification Project")
    
    team_info = """
    ### House & Apartments Classification Model
    #### TEAM MEMBERS
    - Сидоркин Георгий Владимирович РИМ-130908
    - Романова Виктория РИМ-130908
    - Гребнев Никита РИМ-130908
    - Шлёгин Лев Русланович РИМ-130908

    #### Our Project
    """
    display_team_info(team_info)
    
    model_name = st.selectbox("Select Classification Model", ["JuanMa360/room-classification", "other-model-1", "other-model-2"])
    model = load_model(model_name)
    
    loaded_image = load_image()
    
    if loaded_image is not None and st.button('Submit'):
        try:
            x = preprocess_image(loaded_image)
            prediction = model(x)
            st.write("#### Output")
            display_classification_results(prediction)
            if st.button("Save Results"):
                save_results(prediction)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if name == "main":
    main()
