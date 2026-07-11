# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2>=3.1",
# ]
# ///
"""Capture chat-template RENDER-parity references for InternLM3 (Track G).

InternLM3's architecture is numerically parity-proven at 1e-4 by
`capture_internlm3.py` + `InternLM3*ParityTests`. This follow-up verifies that the
checkpoint's OWN `chat_template` (shipped inside `tokenizer_config.json`) renders
correctly under swift-jinja (the engine swift-transformers drives), so NO built-in
chat-template override is needed — the Hunyuan V1 Dense / MiniCPM3 precedent, NOT the
Seed-OSS / Cohere2 one.

InternLM3's template is a plain ChatML wrapper:

    {{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role']
    + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if
    add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}

It uses only constructs swift-jinja fully supports — the message loop, string
concatenation, `add_generation_prompt`, and the tokenizer-injected `bos_token`. There
are no tool-use macros, recursive `{% call %}`, integer-keyed dicts, or other
swift-jinja-hostile constructs, so it renders byte-for-byte on every path.

This script renders the checkpoint's template for representative message sets and
stores the rendered prompt strings (plus the template and the tokenizer's bos/eos
tokens) as a JSON fixture. The Swift test `InternLM3ChatTemplateParityTests` renders
the SAME template through swift-jinja and asserts byte-for-byte equality — proving
native rendering with NO model weights and NO Metal required.

Rendering uses jinja2 with `trim_blocks=True, lstrip_blocks=True`, matching both
(a) how HuggingFace transformers compiles chat templates and (b) how
swift-transformers configures swift-jinja.

WHY NO `transformers.AutoTokenizer` CROSS-CHECK (a deliberate departure from the
Hunyuan / MiniCPM3 capture scripts): the InternLM3 repo ships a CUSTOM tokenizer
class, so `AutoTokenizer.from_pretrained` demands `trust_remote_code=True` — i.e. it
would execute the repo's Python. We decline to run untrusted remote code, and it is
unnecessary here for two reasons: (1) swift-transformers NEVER runs that Python in
production — it drives `tokenizer.json` plus the `chat_template` STRING through
swift-jinja, exactly what this script reads and renders; (2) the template is trivial
ChatML that reads only `bos_token`, `messages`, and `add_generation_prompt`, with no
macros or conditional logic, so jinja2 (the engine transformers itself wraps for
`apply_chat_template`) with the same options and injected `bos_token` IS the
authoritative render. The template + special tokens are read directly from
`tokenizer_config.json`.

Run (checkpoint present in the HF cache, or MACMLX_INTERNLM3_MODEL_DIR set):
    uv run docs/reference/capture_internlm3_chat_template.py
"""
import glob
import json
import os
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


def _token_str(value) -> str:
    """Special tokens in `tokenizer_config.json` are either a bare string or an
    `AddedToken`-style dict carrying a `content` field. Normalize to the string."""
    if isinstance(value, dict):
        return value.get("content", "")
    return value or ""


def load_template_and_tokens(snap: str):
    """Read the chat_template + bos/eos tokens straight from tokenizer_config.json —
    no tokenizer instantiation, so no `trust_remote_code` execution."""
    with open(os.path.join(snap, "tokenizer_config.json"), encoding="utf-8") as f:
        tc = json.load(f)
    template = tc.get("chat_template")
    if template is None:
        raise SystemExit("InternLM3 tokenizer_config.json carries no chat_template")
    return template, _token_str(tc.get("bos_token")), _token_str(tc.get("eos_token"))


def snapshot_dir() -> str:
    """Resolve the InternLM3 checkpoint dir: env override → HF cache snapshot."""
    env = os.environ.get("MACMLX_INTERNLM3_MODEL_DIR")
    if env and os.path.exists(os.path.join(env, "config.json")):
        return env
    home = os.path.expanduser("~")
    cache = os.path.join(
        home,
        ".cache/huggingface/hub/"
        "models--mlx-community--internlm3-8b-instruct-4bit/snapshots",
    )
    for d in sorted(glob.glob(os.path.join(cache, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    raise SystemExit(
        "InternLM3 checkpoint not found. Set MACMLX_INTERNLM3_MODEL_DIR or place it "
        "in the HuggingFace cache."
    )


# Representative message sets. Each exercises a distinct template path:
#   • single_user            — bos + a single user turn + generation prompt.
#   • single_user_no_gen      — the same without add_generation_prompt.
#   • system_user            — a leading system message then a user turn.
#   • multi_turn             — user/assistant/user (a historical assistant turn) then
#                              a trailing user + generation prompt.
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
    )


def main() -> None:
    snap = snapshot_dir()
    original, bos, eos = load_template_and_tokens(snap)

    out_cases = []
    for case in CASES:
        rendered = render_jinja2(original, case, bos, eos)
        out_cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "expected": rendered,
            }
        )
        print(f"case {case['name']}: {len(rendered)} chars (jinja2)")

    fixtures_dir = Path(__file__).resolve().parents[2] / (
        "MacMLXCore/Tests/MacMLXCoreTests/Fixtures"
    )
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_path = fixtures_dir / "internlm3_chat_template_fixture.json"
    payload = {
        "_comment": (
            "Render-parity reference for the InternLM3 checkpoint's OWN chat_template "
            "(shipped in tokenizer_config.json; no macMLX override needed). Generated "
            "by docs/reference/capture_internlm3_chat_template.py (jinja2, trim_blocks "
            "+ lstrip_blocks — the exact engine swift-jinja ports; no transformers "
            "tokenizer instantiation, which the InternLM3 repo gates behind "
            "trust_remote_code). InternLM3ChatTemplateParityTests renders `template` "
            "through swift-jinja with the same bos_token/eos_token and asserts "
            "equality with each case's `expected`, proving swift-jinja renders the "
            "checkpoint's plain-ChatML template natively."
        ),
        "source_repo": "mlx-community/internlm3-8b-instruct-4bit",
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
