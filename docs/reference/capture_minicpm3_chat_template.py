# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2>=3.1",
#     "transformers>=4.41,<5.13",
# ]
# ///
"""Capture chat-template RENDER-parity references for MiniCPM3 (Track G).

MiniCPM3's architecture is numerically parity-proven at 1e-4 by
`capture_minicpm3.py` + `MiniCPM3ModelParityTests`. This follow-up captures the
checkpoint's OWN `chat_template` (stored inline in `tokenizer_config.json`, not a
standalone `chat_template.jinja` file) rendered for representative STANDARD
conversation message sets, so the Swift side can prove render-parity.

The MiniCPM3 template is a heavy tool-use template: recursive Jinja macros
(`json_to_python_type`, `object_to_fields` with `{% call %}`/`caller()`), plus
`|items`, `|tojson`, `|title`, `is iterable`, `is defined`, `namespace`. jinja2
supports all of it; whether swift-jinja can PARSE it is exactly what the Swift test
decides. This script renders the ORIGINAL template with jinja2 (cross-checked ==
transformers `apply_chat_template`) and stores the rendered prompt strings, so the
Swift test is authoritative regardless of which rendering engine ends up applied.

Rendering uses jinja2 with `trim_blocks=True, lstrip_blocks=True`, matching both
(a) how HuggingFace transformers compiles chat templates and (b) how
swift-transformers configures swift-jinja. This script asserts jinja2 ==
transformers' `apply_chat_template` for every case before writing.

Run (checkpoint present in the HF cache, or MACMLX_MINICPM3_MODEL_DIR set):
    uv run docs/reference/capture_minicpm3_chat_template.py
"""
import glob
import json
import os
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


def snapshot_dir() -> str:
    """Resolve the MiniCPM3 checkpoint dir: env override → HF cache snapshot."""
    env = os.environ.get("MACMLX_MINICPM3_MODEL_DIR")
    if env and os.path.exists(os.path.join(env, "config.json")):
        return env
    home = os.path.expanduser("~")
    cache = os.path.join(
        home,
        ".cache/huggingface/hub/"
        "models--mlx-community--MiniCPM3-4B-4bit/snapshots",
    )
    for d in sorted(glob.glob(os.path.join(cache, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    raise SystemExit(
        "MiniCPM3 checkpoint not found. Set MACMLX_MINICPM3_MODEL_DIR or place it "
        "in the HuggingFace cache."
    )


# Representative STANDARD-path message sets (no tools, no tool_calls/thought — those
# branches are off the plain chat path). Each exercises a distinct template path:
#   • single_user            — a single user turn + generation prompt (no system).
#   • single_user_no_gen      — the same without add_generation_prompt.
#   • system_user            — the leading system-message assembly branch.
#   • multi_turn             — user/assistant/user (the plain assistant `else`
#                              branch: role + content + im_end) then a trailing user.
#   • end_assistant_no_gen    — a conversation ending on assistant, add_gen False.
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
]


def render_jinja2(template_str: str, case: dict) -> str:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    tmpl = env.from_string(template_str)
    return tmpl.render(
        messages=case["messages"],
        add_generation_prompt=case["add_generation_prompt"],
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
    with open(os.path.join(snap, "tokenizer_config.json"), encoding="utf-8") as f:
        original = json.load(f)["chat_template"]

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
                "expected": rendered,
            }
        )
        print(f"case {case['name']}: {len(rendered)} chars (jinja2 == transformers)")

    fixtures_dir = Path(__file__).resolve().parents[2] / (
        "MacMLXCore/Tests/MacMLXCoreTests/Fixtures"
    )
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixtures_dir / "minicpm3_chat_template_fixture.json"
    payload = {
        "_comment": (
            "Render-parity reference for the MiniCPM3 checkpoint's OWN chat_template "
            "(inline in tokenizer_config.json). Generated by "
            "docs/reference/capture_minicpm3_chat_template.py (jinja2, trim_blocks + "
            "lstrip_blocks, cross-checked == transformers apply_chat_template). "
            "MiniCPM3ChatTemplateParityTests renders each case through swift-jinja and "
            "asserts equality with `expected`. Only the standard conversation path is "
            "covered (no tools / tool_calls / thought). `template` is the ORIGINAL, "
            "unmodified checkpoint template; if the test targets a built-in override "
            "instead, the override must reproduce these standard-path renders "
            "byte-for-byte."
        ),
        "source_repo": "mlx-community/MiniCPM3-4B-4bit",
        "template": original,
        "cases": out_cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
