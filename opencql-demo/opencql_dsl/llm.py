"""
OpenCQL LLM Layer v2
Supports: Anthropic Claude, OpenAI, Ollama (local), Mock
Provider is selected by model name prefix or explicit provider= kwarg.
"""

from __future__ import annotations
import os
import json
import traceback
from typing import Optional,  Any


# \u2500\u2500 Provider detection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _detect_provider(model_name: str) -> str:
    model_lower = model_name.lower()
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
        return "openai"
    if model_lower in ("llama3", "llama3.2", "mistral", "mixtral", "phi3", "gemma"):
        return "ollama"
    return "mock"


# \u2500\u2500 Base LLM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class LLM:
    def __init__(self, model_name: str = "mock", provider: str | None = None, **kwargs):
        self.model_name = model_name
        self.provider = provider or _detect_provider(model_name)
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        try:
            if self.provider == "anthropic":
                return self._anthropic(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "openai":
                return self._openai(prompt, system_prompt, temperature, max_tokens)
            elif self.provider == "ollama":
                return self._ollama(prompt, system_prompt)
            else:
                return self._mock(prompt, system_prompt)
        except Exception as e:
            traceback.print_exc()
            return f"[LLM Error: {e}]"

    # \u2500\u2500 Anthropic \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _anthropic(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        msg = client.messages.create(**kwargs)
        return msg.content[0].text

    # \u2500\u2500 OpenAI \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _openai(self, prompt, system_prompt, temperature, max_tokens) -> str:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    # \u2500\u2500 Ollama \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _ollama(self, prompt, system_prompt) -> str:
        import requests
        full = f"System: {system_prompt}\
User: {prompt}" if system_prompt else prompt
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model_name, "prompt": full, "stream": False},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        raise RuntimeError(f"Ollama returned {resp.status_code}")

    # \u2500\u2500 Mock (no API key needed) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _mock(self, prompt, system_prompt) -> str:
        snippet = prompt[:120].replace("\
", " ")
        role = system_prompt[:60] if system_prompt else "general assistant"
        return (
            f"[Mock {self.model_name} | role={role}] "
            f"I have analyzed the following context: \"{snippet}...\" "
            f"and produced a structured response."
        )


# \u2500\u2500 Convenience factory \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def create_llm(model_name: str = "mock", **kwargs) -> LLM:
    """
    Factory. Examples:
        create_llm("claude-3-5-sonnet-20241022")
        create_llm("gpt-4o")
        create_llm("llama3")
        create_llm("mock")
    """
    return LLM(model_name=model_name, **kwargs)


# \u2500\u2500 Legacy shim (backwards compat with original opencql) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class OpenSourceLLM(LLM):
    """Drop-in replacement for the original OpenSourceLLM class."""
    def __init__(self, model_name="llama3"):
        super().__init__(model_name=model_name, provider="ollama")
