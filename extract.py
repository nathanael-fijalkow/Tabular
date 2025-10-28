import google.generativeai as genai
import os
import base64
import yaml
from llm import LLM

with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

def convert_image_to_data_uri(image_path: str, mime_type: str) -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    data_uri = f'data:{mime_type};base64,{base64_encoded_data}'
    return data_uri

def OCR_to_markdown(llm: LLM, image_path: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    data_uri = convert_image_to_data_uri(image_path, "image/png")
    prompt = f"Image: {data_uri}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["OCR_SYSTEM_PROMPT"])

    # Construct the Data URI string (includes the MIME type)
    data_uri = f'data:{mime_type};base64,{base64_encoded_data}'
    return data_uri

def OCR_to_markdown(llm: LLM, image_path: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    data_uri = convert_image_to_data_uri(image_path, "image/png")
    prompt = f"Image: {data_uri}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["OCR_SYSTEM_PROMPT"])

def extract_JSON_from_image(llm: LLM, image_path: str, json_schema: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    data_uri = convert_image_to_data_uri(image_path, "image/png")
    prompt = f"Image: {data_uri}\nJSON Schema: {json_schema}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["JSON_EXTRACTION_SYSTEM_PROMPT"])

def extract_image_from_document(llm: LLM, image_path: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    prompt = f"Image bytes: {image_bytes}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["IMAGE_EXTRACTION_SYSTEM_PROMPT"])

