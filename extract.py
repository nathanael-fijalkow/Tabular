import google.generativeai as genai
import os
import yaml
from llm import LLM

with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

def OCR_to_markdown(llm: LLM, image_path: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    prompt = f"Image bytes: {image_bytes}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["OCR_SYSTEM_PROMPT"])

def extract_JSON_from_image(llm: LLM, image_path: str, json_schema: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    prompt = f"Image bytes: {image_bytes}\nJSON Schema: {json_schema}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["JSON_EXTRACTION_SYSTEM_PROMPT"])

def extract_image_from_document(llm: LLM, image_path: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    prompt = f"Image bytes: {image_bytes}"
    return llm.generate(prompt, provider=provider, model_name=model_name, system_prompt=prompts["IMAGE_EXTRACTION_SYSTEM_PROMPT"])

