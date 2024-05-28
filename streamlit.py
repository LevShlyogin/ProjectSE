import io
import streamlit as st
import tensorflow as tf
from PIL import Image
from transformers import pipeline
from typing import Optional, Dict


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