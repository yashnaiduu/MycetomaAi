import hashlib
import json
import logging
import os
from pathlib import Path

from backend.src.utils.config import load_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a medical AI assistant specializing in Mycetoma diagnosis. "
    "Given prediction results from a histopathology image analysis, "
    "provide a brief clinical explanation. Be precise and professional."
)

TEMPLATE = (
    "Prediction: {class_name} (confidence: {confidence:.1%})\n"
    "Subtype: {subtype}\n"
    "Class probabilities: {probabilities}\n\n"
    "Provide a 3-4 sentence clinical explanation."
)


class ExplanationService:
    def __init__(self):
        cfg = load_config("api").get("llm", {})
        self.model = cfg.get("model", "gpt-4o-mini")
        self.temperature = cfg.get("temperature", 0)
        self.max_tokens = cfg.get("max_tokens", 300)

        self.cache_dir = Path(cfg.get("cache_dir", ".cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def _cache_key(self, class_name: str, confidence: float, subtype: str) -> str:
        raw = f"{class_name}:{confidence:.2f}:{subtype}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("explanation")
        return None

    def _set_cached(self, key: str, explanation: str):
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({"explanation": explanation}))

    async def generate(
        self,
        class_name: str,
        confidence: float,
        subtype: str,
        probabilities: dict,
    ) -> dict:
        cache_key = self._cache_key(class_name, confidence, subtype)
        cached = self._get_cached(cache_key)
        if cached:
            return {"explanation": cached, "cached": True}

        if not self.api_key:
            fallback = self._fallback_explanation(class_name, confidence, subtype)
            return {"explanation": fallback, "cached": False}

        prompt = TEMPLATE.format(
            class_name=class_name,
            confidence=confidence,
            subtype=subtype,
            probabilities=json.dumps(probabilities),
        )

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("LLM call failed")
            explanation = self._fallback_explanation(class_name, confidence, subtype)

        self._set_cached(cache_key, explanation)
        return {"explanation": explanation, "cached": False}

    @staticmethod
    def _fallback_explanation(class_name: str, confidence: float, subtype: str) -> str:
        if class_name == "Eumycetoma":
            desc = "fungal infection requiring antifungal therapy and potential surgical intervention"
        elif class_name == "Actinomycetoma":
            desc = "bacterial infection typically responsive to antibiotic combination therapy"
        else:
            desc = "no significant mycetoma pathology detected"

        return (
            f"The histopathological analysis indicates {class_name} "
            f"with {confidence:.1%} confidence. This suggests {desc}. "
            f"Suspected causative organism: {subtype}. "
            f"Clinical correlation and microbiological confirmation recommended."
        )
