import base64
import json
import os
from typing import Annotated, Any, Callable, List, Optional

import firebase_admin
import streamlit as st
from firebase_admin import credentials, storage
from langchain_core.tools import tool
from openai import OpenAI
from PIL import Image, ImageDraw


@tool
def create_image_prompt(
    main_character: str, text: str, items_to_include: List[str], color_palette: str
) -> str:
    """Create a prompt for generating an image with a main character, text, items, and a color palette.

    Args:
        main_character (str): The main character to be featured in the image.
        text (str): Text to be included in the image.
        items_to_include (List[str]): List of items to include in the image.
        color_palette (str): Color palette to be used in the image.

    Returns:
        str: A formatted prompt for image generation.
    """
    main_character_prompt = f"Diseña una obra de arte audaz y exagerada, con {main_character} como personaje central."

    items_prompt = ""
    if items_to_include:
        items_prompt = "El diseño debe incluir: " + ", ".join(items_to_include) + "."

    text_prompt = ""
    if text:
        text_prompt = (
            f"El texto '{text}' DEBE APARECER ESCRITO de forma clara en el diseño."
        )

    style_prompt = f"""<ESTILO>
        Genera un diseño estilo caricaturesco y exagerado con un enfoque llamativo y vibrante.
        Usa un personaje central de apariencia humorística, con rasgos detallados y expresiones exageradas.
        Presentar un enfoque que emplea principalmente el blanco y el negro y los colores {color_palette} para añadir contraste.
        **EL FONDO DEBE SER NEGRO**
        Los objetos deben tener un estilo hiperrealista con detalles texturizados y reflejos luminosos.
        La tipografía debe tener un aire vintage, reminiscente de carteles clásicos.
        </ESTILO>
    """

    # Se reorganiza el prompt final para dar prioridad a las instrucciones de formato.
    return f"""
        <OBJETIVO>
        {main_character_prompt}
        </OBJETIVO>

        <INSTRUCCIONES>
        {text_prompt}
        {items_prompt}
         **La palabra 'OWNIT' debe estar integrada sutilmente en el diseño, de forma que no sea el foco principal.**
        </INSTRUCCIONES>

        <OBSERVACIONES>
        </OBSERVACIONES>

        {style_prompt}
    """


@tool
def create_image(
    prompt: Annotated[str, "Prompt IN SPANISH summarizing the user desires"],
    image_number: int,
    output_path: Annotated[
        Optional[str], "Optional full path for the output image."
    ] = None,
) -> str:
    """Create an image to be used as a streetwear shirt design.

    Use this function to produce an image for a tshirt based on the user query.
    You must produce a query that summarizes well the user desires.

    Args:
        query (str): Query IN SPANISH summarizing the user desires.
        image_number (int): The number of the image to be created, used to name the file.

    Returns:
        str: A message indicating the path to the image.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.images.generate(
        model="dall-e-3",
        # background="transparent", ONLY FOR gpt-image-1
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json",
        style="vivid",
        quality="standard",
        n=1,
    )

    image_b64 = response.data[0].b64_json
    image_data = base64.b64decode(image_b64)

    if not output_path:
        output_path = f"image-{image_number}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(image_data)

    return output_path


def upload_to_firebase(file_path: str) -> str | None:
    """Uploads a file to Firebase Storage."""
    try:
        # 1. Initialize Firebase (if not already done)
        if not firebase_admin._apps:
            cred = credentials.Certificate(st.secrets["firebase_service_account"])
            firebase_admin.initialize_app(
                cred, {"storageBucket": st.secrets["FIREBASE_STORAGE_BUCKET"]}
            )

        bucket = storage.bucket()
        file_name = os.path.basename(file_path)
        # Create a blob and upload the file
        blob = bucket.blob(f"images/{file_name}")
        blob.upload_from_filename(file_path)

        # 4. Make the file public and get the URL
        blob.make_public()
        print(f"File {file_name} uploaded, View Link: {blob.public_url}")
        return blob.public_url

    except Exception as e:
        print(f"Failed to upload to Firebase: {e}")
        return None


@tool
def convert_black_to_transparent(
    image_path: Annotated[str, "Path to the image to be converted"],
    output_path: Annotated[str, "Path to save the converted image"],
) -> str:
    """Convert dark pixels in an image to transparent.

    Args:
        image_path (str): Path to the image to be converted.
        color (str): Color to be made transparent in the format 'R,G,B'.

    Returns:
        str: A message indicating the path to the converted image.
    """
    threshold = 30
    img = Image.open(image_path).convert("RGBA")
    pixel_data = img.getdata()

    new_pixel_data = []
    for item in pixel_data:
        r, g, b, a = item
        brightness = (r + g + b) / 3
        if brightness < threshold:
            new_pixel_data.append((r, g, b, 0))
        else:
            new_pixel_data.append((r, g, b, a))

    img.putdata(new_pixel_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


TOOLS: List[Callable[..., Any]] = [
    create_image_prompt,
    create_image,
    convert_black_to_transparent,
]
