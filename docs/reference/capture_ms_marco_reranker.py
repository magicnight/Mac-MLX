"""Reference scores for cross-encoder/ms-marco-MiniLM-L-6-v2 (PyTorch, fp32, CPU).

Feeds `MacMLXCore/Tests/MacMLXCoreTests/Fixtures/rerank_ms_marco_reference.json`,
which `RerankEngineSmokeTests` compares the MLX reranker against on the logit scale.

  uv run --with torch --with transformers --with safetensors \
    python docs/reference/capture_ms_marco_reranker.py \
    > MacMLXCore/Tests/MacMLXCoreTests/Fixtures/rerank_ms_marco_reference.json

Prints raw logits and sigmoid(logit) per pair, plus the token ids and token_type_ids
the HF tokenizer produced (recorded for inspection; the Swift test compares logits). The
checkpoint is fetched from (or found in) the Hugging Face cache; leave
HF_HUB_OFFLINE unset — a partial cache without `.no_exist` markers makes
transformers refuse to load offline.
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"  # pinned so the fixture is reproducible
QUERY = "How many people live in Berlin?"
DOCS = [
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin is the capital of Germany and about 3.5 million people live there.",
    "The recipe calls for two eggs, a cup of flour and a pinch of salt.",
]
import transformers
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, revision=REVISION, torch_dtype=torch.float32).eval()
out = {
    "model": MODEL,
    "revision": REVISION,
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    "query": QUERY,
    "pairs": [],
}
with torch.no_grad():
    for doc in DOCS:
        enc = tok(QUERY, doc, return_tensors="pt", truncation=True, max_length=512)
        logit = model(**enc).logits[0, 0].item()
        out["pairs"].append({
            "document": doc,
            "input_ids": enc["input_ids"][0].tolist(),
            "token_type_ids": enc["token_type_ids"][0].tolist(),
            "logit": logit,
            "sigmoid": 1.0 / (1.0 + torch.exp(torch.tensor(-logit)).item()),
        })
    # one batched pass too, to show padding does not change the numbers
    enc = tok([QUERY] * len(DOCS), DOCS, return_tensors="pt", padding=True, truncation=True, max_length=512)
    out["batched_logits"] = model(**enc).logits[:, 0].tolist()
print(json.dumps(out, indent=1))
