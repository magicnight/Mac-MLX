"""Reference scores for cross-encoder/nli-MiniLM2-L6-H768 (PyTorch, fp32, CPU).

Feeds `MacMLXCore/Tests/MacMLXCoreTests/Fixtures/rerank_nli_reference.json`, which
`RerankEngineSmokeTests` compares the MLX reranker against: a 3-way NLI head that
MLXRerankers scores as the softmax probability of the `entailment` class.

  uv run --with torch --with transformers --with safetensors \
    python docs/reference/capture_nli_reranker.py \
    > MacMLXCore/Tests/MacMLXCoreTests/Fixtures/rerank_nli_reference.json

Uses the same query and passages as the ms-marco fixture. Prints the three class
logits per pair, the softmax probabilities, the index of the entailment class and
the HF pair encoding (`<s> q </s></s> d </s>` for a RoBERTa tokenizer), so a
tokenizer discrepancy can be told apart from a model one. Leave HF_HUB_OFFLINE
unset: a partial cache without `.no_exist` markers makes transformers refuse to
load offline.
"""
import json
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"  # pinned so the fixture is reproducible
QUERY = "How many people live in Berlin?"
DOCS = [
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin is the capital of Germany and about 3.5 million people live there.",
    "The recipe calls for two eggs, a cup of flour and a pinch of salt.",
]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, revision=REVISION, torch_dtype=torch.float32).eval()
labels = {int(k): v for k, v in model.config.id2label.items()}
entailment = next(i for i, name in labels.items() if name.lower() == "entailment")
out = {
    "model": MODEL,
    "revision": REVISION,
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    "labels": labels,
    "positive_class": entailment,
    "query": QUERY,
    "pairs": [],
}
with torch.no_grad():
    for doc in DOCS:
        enc = tok(QUERY, doc, return_tensors="pt", truncation=True, max_length=512)
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        out["pairs"].append({
            "document": doc,
            "input_ids": enc["input_ids"][0].tolist(),
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "positive_probability": probs[entailment].item(),
        })
print(json.dumps(out, indent=1))
