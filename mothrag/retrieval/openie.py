# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""OpenIE client (NER + triple extraction) via an OpenAI-compatible LLM endpoint.

Replicates HippoRAG's 1-shot NER + triple extraction prompts, calling the LLM
through any OpenAI-compatible API (Together AI, OpenAI, vLLM, etc.). No
HippoRAG package installation is required — only the prompt templates are
reused.

Default endpoint: Together AI's Llama-3.3-70B-Instruct-Turbo.

Usage::

    from mothrag.retrieval.openie import OpenIEClient
    client = OpenIEClient(api_key=os.getenv("TOGETHER_API_KEY"))
    result = client.process("Some Wikipedia paragraph...")
    print(result.entities)
    print(result.triples)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from mothrag.utils.url_safety import validate_base_url


# --- Prompt templates (verbatim from HippoRAG src/hipporag/prompts/templates/) ---

NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
"""

ONE_SHOT_NER_PARAGRAPH = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

ONE_SHOT_NER_OUTPUT = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

TRIPLE_SYSTEM = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

TRIPLE_FRAME = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

ONE_SHOT_TRIPLE_INPUT = TRIPLE_FRAME.format(
    passage=ONE_SHOT_NER_PARAGRAPH, named_entity_json=ONE_SHOT_NER_OUTPUT
)

ONE_SHOT_TRIPLE_OUTPUT = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""


# --- Response parsers ---

def _extract_ner_entities(response_text: str) -> list[str]:
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, response_text, re.DOTALL)
    if match is None:
        return []
    try:
        return json.loads(match.group()).get("named_entities", [])
    except json.JSONDecodeError:
        return []


def _extract_triples(response_text: str) -> list[list[str]]:
    pattern = r'\{[^{}]*"triples"\s*:\s*\[\s*(?:\[[^\]]*\]\s*,?\s*)*\][^{}]*\}'
    match = re.search(pattern, response_text, re.DOTALL)
    if match is None:
        return []
    try:
        out = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    raw_triples = out.get("triples", [])
    return [t for t in raw_triples if isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t)]


# --- Client ---

@dataclass
class OpenIEResult:
    entities: list[str]
    triples: list[list[str]]
    chunk_id: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0


class OpenIEClient:
    """Minimal OpenIE client using any OpenAI-compatible endpoint.

    Default: ``meta-llama/Llama-3.3-70B-Instruct-Turbo`` via Together AI.
    For OpenAI: pass ``base_url=None`` and ``api_key=$OPENAI_API_KEY``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.together.xyz/v1",
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        *,
        allow_custom_endpoint: bool = False,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package not installed. pip install openai") from e
        if api_key is None:
            api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not provided. Set TOGETHER_API_KEY or pass api_key=...")
        # Allowlist guard for base_url. Default Together AI URL is allowlisted.
        validated_base_url = validate_base_url(
            base_url, allow_custom_endpoint=allow_custom_endpoint,
        )
        self.client = OpenAI(api_key=api_key, base_url=validated_base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _chat(self, messages: list[dict], max_tokens: int = 1024,
              temperature: float = 0.0) -> tuple[str, int, int, float]:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                latency = time.time() - t0
                content = resp.choices[0].message.content or ""
                tokens_in = resp.usage.prompt_tokens if resp.usage else 0
                tokens_out = resp.usage.completion_tokens if resp.usage else 0
                return content, tokens_in, tokens_out, latency
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"OpenIE chat failed after {self.max_retries} retries: {last_err}")

    def extract_entities(self, passage: str, chunk_id: str = "anon") -> OpenIEResult:
        messages = [
            {"role": "system", "content": NER_SYSTEM},
            {"role": "user", "content": ONE_SHOT_NER_PARAGRAPH},
            {"role": "assistant", "content": ONE_SHOT_NER_OUTPUT},
            {"role": "user", "content": passage},
        ]
        content, t_in, t_out, lat = self._chat(messages, max_tokens=512)
        entities = _extract_ner_entities(content)
        unique = list(dict.fromkeys(entities))
        return OpenIEResult(
            entities=unique, triples=[], chunk_id=chunk_id,
            tokens_in=t_in, tokens_out=t_out, latency_s=lat,
        )

    def extract_triples(self, passage: str, entities: list[str],
                        chunk_id: str = "anon") -> OpenIEResult:
        named_entity_json = json.dumps({"named_entities": entities}, ensure_ascii=False)
        user_msg = TRIPLE_FRAME.format(passage=passage, named_entity_json=named_entity_json)
        messages = [
            {"role": "system", "content": TRIPLE_SYSTEM},
            {"role": "user", "content": ONE_SHOT_TRIPLE_INPUT},
            {"role": "assistant", "content": ONE_SHOT_TRIPLE_OUTPUT},
            {"role": "user", "content": user_msg},
        ]
        content, t_in, t_out, lat = self._chat(messages, max_tokens=2048)
        triples = _extract_triples(content)
        return OpenIEResult(
            entities=entities, triples=triples, chunk_id=chunk_id,
            tokens_in=t_in, tokens_out=t_out, latency_s=lat,
        )

    def process(self, passage: str, chunk_id: str = "anon") -> OpenIEResult:
        """Full pipeline: NER -> triples. Returns merged result."""
        ner_res = self.extract_entities(passage, chunk_id)
        if not ner_res.entities:
            return ner_res
        tr_res = self.extract_triples(passage, ner_res.entities, chunk_id)
        return OpenIEResult(
            entities=ner_res.entities,
            triples=tr_res.triples,
            chunk_id=chunk_id,
            tokens_in=ner_res.tokens_in + tr_res.tokens_in,
            tokens_out=ner_res.tokens_out + tr_res.tokens_out,
            latency_s=ner_res.latency_s + tr_res.latency_s,
        )
