# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2>=3.1",
#     "transformers>=4.55,<5.13",
# ]
# ///
"""Capture chat-template RENDER-parity references for Seed-OSS (Track G follow-up).

The Seed-OSS architecture is numerically parity-proven at 1e-4 by
`capture_seed_oss.py` + `SeedOss*ParityTests`, but end-to-end generation was
blocked because the checkpoint's `chat_template.jinja` builds its thinking-budget
table as a Jinja object literal with INTEGER keys, which swift-jinja 2.3.6 cannot
parse. macMLX ships a built-in override (`SeedOssChatTemplate.swift`) that
rewrites ONLY that one construct as an equivalent if/elif ladder.

This script renders the checkpoint's ORIGINAL template (the reference
implementation) for a few representative message sets and stores the rendered
prompt strings as a JSON fixture. The Swift test `SeedOssChatTemplateParityTests`
renders the SAME message sets through the OVERRIDE template via swift-jinja and
asserts byte-for-byte string equality — proving the rewrite is semantically
identical, with NO model weights and NO Metal required.

Rendering uses jinja2 with `trim_blocks=True, lstrip_blocks=True`, matching both
(a) how HuggingFace transformers compiles chat templates and (b) how
swift-transformers configures swift-jinja (`Template(_, with: .init(lstripBlocks:
true, trimBlocks: true))`). This script asserts jinja2 == transformers'
`apply_chat_template` for every case before writing, so the reference is
authoritative.

Run (checkpoint present in the HF cache, or MACMLX_SEED_OSS_MODEL_DIR set):
    uv run docs/reference/capture_seed_oss_chat_template.py
"""
import glob
import json
import os
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


def snapshot_dir() -> str:
    """Resolve the Seed-OSS checkpoint dir: env override → HF cache snapshot."""
    env = os.environ.get("MACMLX_SEED_OSS_MODEL_DIR")
    if env and os.path.exists(os.path.join(env, "config.json")):
        return env
    home = os.path.expanduser("~")
    cache = os.path.join(
        home,
        ".cache/huggingface/hub/"
        "models--mlx-community--Seed-OSS-36B-Instruct-4bit/snapshots",
    )
    for d in sorted(glob.glob(os.path.join(cache, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    raise SystemExit(
        "Seed-OSS checkpoint not found. Set MACMLX_SEED_OSS_MODEL_DIR or place it "
        "in the HuggingFace cache."
    )


# Representative message sets. Each exercises a distinct template path:
#   • system_user     — the leading-system branch + user/system message loop.
#   • multi_turn      — assistant history (the `role == "assistant"` branch).
#   • thinking_budget — the budget-engaged system block, i.e. the exact code path
#                       whose reflection-interval lookup was rewritten.
CASES = [
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
        "thinking_budget": None,
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke."},
        ],
        "add_generation_prompt": True,
        "thinking_budget": None,
    },
    {
        "name": "thinking_budget_512",
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "add_generation_prompt": True,
        "thinking_budget": 512,
    },
]

# The exact construct this override rewrites is the thinking-budget
# reflection-interval lookup: the upstream template does
# `budget_reflections_v05 = {0:0, 512:128, 1024:256, 2048:512, 4096:512,
# 8192:1024, 16384:1024}` then `dictsort` + "first tier whose key >=
# thinking_budget" (with a `[16384]` fallback past the top tier). Every tier
# boundary is swept ±1 to pin the < / <= edge exactly, plus 0 (its own
# skip-thinking branch, no interval), a negative budget (the "no budget"
# default branch, no interval), and one past-top value. This is the ORIGINAL
# template's rendering for each — the Swift test asserts the OVERRIDE renders
# identically, closing the loop so the boundary proof is against the real
# reference rather than a hand-copied table.
BOUNDARY_BUDGETS = [
    511, 512, 513,
    1023, 1024, 1025,
    2047, 2048, 2049,
    4095, 4096, 4097,
    8191, 8192, 8193,
    16383, 16384, 16385,
    0, -1, 100000,
]

BOUNDARY_MESSAGES = [{"role": "user", "content": "hi"}]


def render_jinja2(template_str: str, case: dict) -> str:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    tmpl = env.from_string(template_str)
    ctx = {
        "messages": case["messages"],
        "add_generation_prompt": case["add_generation_prompt"],
    }
    if case["thinking_budget"] is not None:
        ctx["thinking_budget"] = case["thinking_budget"]
    return tmpl.render(**ctx)


def render_transformers(snap: str, case: dict) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    kwargs = {"add_generation_prompt": case["add_generation_prompt"]}
    if case["thinking_budget"] is not None:
        kwargs["thinking_budget"] = case["thinking_budget"]
    return tok.apply_chat_template(case["messages"], tokenize=False, **kwargs)


def main() -> None:
    snap = snapshot_dir()
    with open(os.path.join(snap, "chat_template.jinja"), encoding="utf-8") as f:
        original = f.read()

    out_cases = []
    for case in CASES:
        rendered = render_jinja2(original, case)
        # Cross-check jinja2 against transformers' own apply_chat_template so the
        # stored reference is authoritative (not merely "some jinja render").
        reference = render_transformers(snap, case)
        if rendered != reference:
            raise SystemExit(
                f"jinja2 vs transformers mismatch for case {case['name']!r}:\n"
                f"  jinja2:       {rendered!r}\n"
                f"  transformers: {reference!r}"
            )
        out_cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "thinking_budget": case["thinking_budget"],
                "expected": rendered,
            }
        )
        print(f"case {case['name']}: {len(rendered)} chars (jinja2 == transformers)")

    out_boundary_cases = []
    for budget in BOUNDARY_BUDGETS:
        case = {
            "name": f"boundary_{budget}",
            "messages": BOUNDARY_MESSAGES,
            "add_generation_prompt": True,
            "thinking_budget": budget,
        }
        rendered = render_jinja2(original, case)
        reference = render_transformers(snap, case)
        if rendered != reference:
            raise SystemExit(
                f"jinja2 vs transformers mismatch for boundary budget {budget!r}:\n"
                f"  jinja2:       {rendered!r}\n"
                f"  transformers: {reference!r}"
            )
        out_boundary_cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "thinking_budget": case["thinking_budget"],
                "expected": rendered,
            }
        )
        print(f"boundary budget={budget}: {len(rendered)} chars (jinja2 == transformers)")

    fixtures_dir = Path(__file__).resolve().parents[2] / (
        "MacMLXCore/Tests/MacMLXCoreTests/Fixtures"
    )
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixtures_dir / "seed_oss_chat_template_fixture.json"
    payload = {
        "_comment": (
            "Render-parity reference for the Seed-OSS chat-template override. "
            "Generated by docs/reference/capture_seed_oss_chat_template.py from "
            "the ORIGINAL checkpoint chat_template.jinja (jinja2, trim_blocks + "
            "lstrip_blocks, cross-checked == transformers apply_chat_template). "
            "SeedOssChatTemplateParityTests renders the OVERRIDE via swift-jinja "
            "and asserts equality with `expected`. `cases` are representative "
            "message-set diversity; `boundary_cases` sweep every thinking-budget "
            "reflection-interval tier boundary (±1) plus 0/negative/past-top, "
            "proving the rewritten if/elif ladder against the ORIGINAL template's "
            "own dict+dictsort lookup rather than a hand-copied table."
        ),
        "source_repo": "mlx-community/Seed-OSS-36B-Instruct-4bit",
        "cases": out_cases,
        "boundary_cases": out_boundary_cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
