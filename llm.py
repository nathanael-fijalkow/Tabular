from __future__ import annotations
from typing import Optional

import google.generativeai as genai
from huggingface_hub import InferenceClient

from config import get_settings

class LLM:
    def __init__(self):
        self.settings = get_settings()

    def generate(self, prompt: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash", system_prompt: Optional[str] = None) -> str:
        if provider == "gemini":
            return self._gemini_generate(prompt, model_name, system_prompt)
        elif provider == "huggingface":
            return self._hf_generate(prompt, model_name, system_prompt)
        else:
            raise ValueError("Unsupported LLM_PROVIDER")

    def _gemini_generate(self, prompt: str, model_name: str = "gemini-2.5-flash", system_prompt: Optional[str] = None) -> str:
        api_key = self.settings.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    def _hf_generate(self, prompt: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", system_prompt: Optional[str] = None) -> str:
        if not self.settings.hf_api_key:
            raise RuntimeError("HF_API_KEY not set")
        client = InferenceClient(
            model=model_name,
            token=self.settings.hf_api_key
        )
        messages = []    
        # Add the system prompt first, if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    
        # Add the user prompt
        messages.append({"role": "user", "content": prompt})
        
                # Use chat_completion instead of chat
        response = client.chat_completion(messages=messages)
        
        content = response.choices[0].message.content
        return content.strip() if content else ""