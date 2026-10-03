# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2>=3.1",
#     "transformers>=4.41,<5.13",
# ]
# ///
"""Capture chat-template RENDER-parity references for Hunyuan V1 Dense (Track G).

Hunyuan V1 Dense's architecture is numerically parity-proven at 1e-4 by
`capture_hunyuan_v1_dense.py` + `HunyuanV1Dense*ParityTests`. This follow-up
verifies that the checkpoint's own `chat_template.jinja` renders correctly under
swift-jinja (the engine swift-transformers drives), so NO built-in chat-template
override is needed — unlike Seed-OSS, whose integer-keyed dict forced one.

The Hunyuan template uses only constructs swift-jinja supports (namespace, the
message loop, string concatenation, `in`, `loop.last`/`loop.index0`, and
tokenizer-injected `bos_token`/`eos_token`). Historically the one construct
swift-jinja rendered differently was
`content.split('<answer>')[-1].strip('</answer>').strip()` (swift-jinja 2.3.6's
`.strip()` ignored its argument and trimmed whitespace only), which sits behind
`'<answer>' in content and not loop.last` — a HISTORICAL assistant turn that
embeds `<answer>` tags. swift-jinja 2.4.0 fixes `strip(arg)` argument handling
(huggingface/swift-jinja #64, reported by macMLX), so that branch now renders
byte-for-byte too; the `answer_history` case below exercises it and the Swift
parity test asserts it.

This script renders the checkpoint's ORIGINAL template for representative message
sets and stores the rendered prompt strings (plus the template and the tokenizer's
bos/eos tokens) as a JSON fixture. The Swift test
`HunyuanV1DenseChatTemplateParityTests` renders the SAME template through
swift-jinja and asserts byte-for-byte equality — proving native rendering with NO
model weights and NO Metal required.

Rendering uses jinja2 with `trim_blocks=True, lstrip_blocks=True`, matching both
(a) how HuggingFace transformers compiles chat templates and (b) how
swift-transformers configures swift-jinja. This script asserts jinja2 ==
transformers' `apply_chat_template` for every case before writing, so the
reference is authoritative.

Run (checkpoint present in the HF cache, or MACMLX_HUNYUAN_V1_MODEL_DIR set):
    uv run docs/reference/capture_hunyuan_v1_dense_chat_template.py
"""
import glob
import json
import os
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


def snapshot_dir() -> str:
    """Resolve the Hunyuan checkpoint dir: env override → HF cache snapshot."""
    env = os.environ.get("MACMLX_HUNYUAN_V1_MODEL_DIR")
    if env and os.path.exists(os.path.join(env, "config.json")):
        return env
    home = os.path.expanduser("~")
    cache = os.path.join(
        home,
        ".cache/huggingface/hub/"
        "models--mlx-community--Hunyuan-1.8B-Instruct-4bit/snapshots",
    )
    for d in sorted(glob.glob(os.path.join(cache, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    raise SystemExit(
        "Hunyuan checkpoint not found. Set MACMLX_HUNYUAN_V1_MODEL_DIR or place it "
        "in the HuggingFace cache."
    )


# Representative message sets. Each exercises a distinct template path:
#   • single_user            — bos + a single user turn + generation prompt.
#   • single_user_no_gen      — the same without add_generation_prompt.
#   • system_user            — the leading system_prompt assembly branch.
#   • two_system             — two system messages concatenated with '\n\n'.
#   • multi_turn             — user/assistant/user (the plain assistant branch,
#                              content + eos_token) then a trailing user.
#   • end_assistant_no_gen    — a conversation ending on assistant, add_gen False
#                              (is_last_user false, no trailing assistant prompt).
#   • answer_history         — a NON-LAST assistant turn embedding <answer></answer>
#                              tags, exercising the off-generation-path
#                              `content.split('<answer>')[-1].strip('</answer>').strip()`
#                              branch that swift-jinja 2.4.0 renders natively (#64).
CASES = [
    {
        "name": "single_user",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "single_user_no_gen",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": False,
    },
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there."},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "two_system",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "system", "content": "You are polite."},
            {"role": "user", "content": "Hi."},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke."},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "end_assistant_no_gen",
        "messages": [
            {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ],
        "add_generation_prompt": False,
    },
    {
        # NON-LAST assistant turn embedding <answer></answer>: triggers the
        # `'<answer>' in content and not loop.last` branch, whose
        # `content.split('<answer>')[-1].strip('</answer>').strip()` keeps only the
        # post-<answer> text with the closing tag's characters stripped. Result for
        # this turn: "The capital of France is Paris." swift-jinja 2.4.0 renders
        # strip(arg) with Python semantics (huggingface/swift-jinja #64).
        "name": "answer_history",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "Let me think about this.<answer>The capital of France is Paris.</answer>",
            },
            {"role": "user", "content": "And what about Germany?"},
        ],
        "add_generation_prompt": True,
    },
]


def render_jinja2(template_str: str, case: dict, bos: str, eos: str) -> str:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    tmpl = env.from_string(template_str)
    return tmpl.render(
        messages=case["messages"],
        add_generation_prompt=case["add_generation_prompt"],
        bos_token=bos,
        eos_token=eos,
        tools=None,
    )


def render_transformers(snap: str, case: dict) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    return tok.apply_chat_template(
        case["messages"],
        tokenize=False,
        add_generation_prompt=case["add_generation_prompt"],
    )


def main() -> None:
    snap = snapshot_dir()
    with open(os.path.join(snap, "chat_template.jinja"), encoding="utf-8") as f:
        original = f.read()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    bos, eos = tok.bos_token, tok.eos_token

    out_cases = []
    for case in CASES:
        rendered = render_jinja2(original, case, bos, eos)
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
                "expected": rendered,
            }
        )
        print(f"case {case['name']}: {len(rendered)} chars (jinja2 == transformers)")

    fixtures_dir = Path(__file__).resolve().parents[2] / (
        "MacMLXCore/Tests/MacMLXCoreTests/Fixtures"
    )
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixtures_dir / "hunyuan_v1_dense_chat_template_fixture.json"
    payload = {
        "_comment": (
            "Render-parity reference for the Hunyuan V1 Dense checkpoint's OWN "
            "chat_template.jinja (no macMLX override needed). Generated by "
            "docs/reference/capture_hunyuan_v1_dense_chat_template.py (jinja2, "
            "trim_blocks + lstrip_blocks, cross-checked == transformers "
            "apply_chat_template). HunyuanV1DenseChatTemplateParityTests renders "
            "`template` through swift-jinja with the same bos_token/eos_token and "
            "asserts equality with each case's `expected`, proving swift-jinja "
            "renders the checkpoint template natively. Covers the standard "
            "conversation path plus the `answer_history` case — a non-last "
            "`<answer>`-tagged assistant turn exercising "
            "content.split('<answer>')[-1].strip('</answer>').strip(), which "
            "swift-jinja 2.4.0 renders with Python strip(arg) semantics "
            "(huggingface/swift-jinja #64)."
        ),
        "source_repo": "mlx-community/Hunyuan-1.8B-Instruct-4bit",
        "template": original,
        "bos_token": bos,
        "eos_token": eos,
        "cases": out_cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
