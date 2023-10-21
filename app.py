import streamlit as st

# Project Title
st.title("Room Classification Project")

# Team Member
st.write("""
         House & Apartaments Classification model
         #### TEAM MEMBER
         - Сидоркин Георгий Владимирович РИМ-130908
         - Рахарди Сандикха РИМ-130908
         -
         -
         
         #### Our Project
         """)

# Upload the image sam
uploadedFile = st.file_uploader('Upload image here')

if uploadedFile is not None:
    st.write("Filename:", uploadedFile.name)
    st.write("Data:", uploadedFile)