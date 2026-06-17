# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Zero-setup MothRAG quickstart.

Runs out of the box on a list of strings — no corpus preprocessing, no data
files. It degrades gracefully with no API keys (hash-embedder + echo-reader
fallbacks), so this script always runs; set keys for production quality:

    pip install 'mothrag[gemini,openai]'
    export GROQ_API_KEY=...      # reader (Llama-3.3-70B)
    export GEMINI_API_KEY=...    # embedder + grounding judge

For a benchmark-style run over a preprocessed corpus, see 01_hotpotqa_eval.py.
"""

from mothrag import MothRAG

if __name__ == "__main__":
    rag = MothRAG.from_documents([
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
        "The Louvre is a museum in Paris.",
    ])

    result = rag.query("In which country is the Eiffel Tower?")

    print("Answer:    ", result.answer)
    print("Arm used:  ", result.arm_used)        # which reasoning arm won arbitration
    print("Confidence:", result.confidence)      # populated when a real reader is configured
