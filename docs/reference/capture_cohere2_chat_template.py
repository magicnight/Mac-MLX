# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2>=3.1",
#     "transformers>=5.0,<5.13",
# ]
# ///
"""Capture chat-template RENDER-parity references for Cohere2 / Command R7B (Track G).

Cohere2's architecture is numerically parity-proven at 1e-4 by
`capture_cohere2.py` + `Cohere2ModelParityTests`. This follow-up verifies that the
checkpoint's OWN `default` chat template renders correctly under swift-jinja (the
engine swift-transformers drives), so NO built-in chat-template override is needed.

WHY THIS ONCE NEEDED AN OVERRIDE: the Command R7B `chat_template` is a LIST of three
NAMED templates (`default` / `tool_use` / `rag`); transformers selects `default`
when neither `tools` nor `documents` is supplied. The `default` template embeds a
large tool/RAG branch behind `{% if documents %}` (with helper macros), and
swift-jinja 2.3.6 could not PARSE that branch ("Unexpected token type:
closeExpression", a literal `}}` misparse), so the whole template failed to compile
even though the branch is off the standard path. That once forced a built-in
override; swift-jinja 2.4.0 fixes the parse (huggingface/swift-jinja #63, reported
by macMLX), so the full `default` template compiles and the standard path renders
natively.

This script renders the ORIGINAL `default` template — the authoritative reference —
for representative message sets, and stores the rendered prompt strings (plus the
template) as a JSON fixture. The Swift test `Cohere2ChatTemplateParityTests` renders
that same `default` template through swift-jinja and asserts byte-for-byte equality
with these references, proving native rendering with NO model weights and NO Metal
required. The STANDARD path uses only swift-jinja-supported constructs (`namespace`,
slicing, boolean comparisons, `content.strip()` with no argument, string
concatenation) and references neither `bos_token` nor `eos_token`.

Rendering uses jinja2 with `trim_blocks=True, lstrip_blocks=True,
keep_trailing_newline=True`, matching both (a) how HuggingFace transformers
compiles chat templates and (b) how swift-transformers configures swift-jinja.
This script asserts jinja2 == transformers' `apply_chat_template` for every case
before writing, so the reference is authoritative.

Run (checkpoint config present in the HF cache, or MACMLX_COHERE2_MODEL_DIR set):
    uv run docs/reference/capture_cohere2_chat_template.py
"""
import glob
import json
import os
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


def snapshot_dir() -> str:
    """Resolve the Cohere2 checkpoint dir: env override → HF cache snapshot.

    Only `tokenizer_config.json` (the chat_template source) is needed, so a
    config-only download of the snapshot suffices.
    """
    env = os.environ.get("MACMLX_COHERE2_MODEL_DIR")
    if env and os.path.exists(os.path.join(env, "tokenizer_config.json")):
        return env
    home = os.path.expanduser("~")
    cache = os.path.join(
        home,
        ".cache/huggingface/hub/"
        "models--mlx-community--c4ai-command-r7b-12-2024-4bit/snapshots",
    )
    for d in sorted(glob.glob(os.path.join(cache, "*"))):
        if os.path.exists(os.path.join(d, "tokenizer_config.json")):
            return d
    raise SystemExit(
        "Cohere2 checkpoint not found. Set MACMLX_COHERE2_MODEL_DIR or place it "
        "in the HuggingFace cache."
    )


def default_template(snap: str) -> str:
    """Extract the `default` named template from the Command R7B chat_template list."""
    with open(os.path.join(snap, "tokenizer_config.json"), encoding="utf-8") as f:
        tc = json.load(f)
    ct = tc["chat_template"]
    if isinstance(ct, list):
        for entry in ct:
            if entry.get("name") == "default":
                return entry["template"]
        raise SystemExit("no `default` template in the chat_template list")
    return ct


# Representative STANDARD-path message sets (no tools, no documents — those
# branches are off the generation path). Each exercises a distinct template path,
# and every case keeps strict user/assistant alternation so the template's
# `raise_exception('Conversation roles must alternate ...')` guard is never tripped:
#   • single_user            — empty system turn + a single user turn + gen prompt.
#   • single_user_no_gen      — the same without add_generation_prompt.
#   • system_user            — a leading system message (system_message branch).
#   • multi_turn             — user/assistant/user (the assistant branch) + gen.
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
    # `jinja2.ext.loopcontrols` mirrors transformers' chat-template environment —
    # the Command R7B `default` template uses `{% break %}` in its (off-path)
    # tool/RAG branch, which is a loopcontrols construct.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    tmpl = env.from_string(template_str)
    return tmpl.render(
        messages=case["messages"],
        add_generation_prompt=case["add_generation_prompt"],
        documents=None,
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
    original = default_template(snap)

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
    out_path = fixtures_dir / "cohere2_chat_template_fixture.json"
    payload = {
        "_comment": (
            "Render-parity reference for the Cohere2 / Command R7B checkpoint's OWN "
            "`default` chat template (no macMLX override needed). Generated by "
            "docs/reference/capture_cohere2_chat_template.py (jinja2, trim_blocks + "
            "lstrip_blocks + keep_trailing_newline, cross-checked == transformers "
            "apply_chat_template). Cohere2ChatTemplateParityTests renders `template` "
            "through swift-jinja and asserts equality with each case's `expected`, "
            "proving swift-jinja parses the whole named `default` template (incl. the "
            "off-path tool/RAG macros) and renders the standard conversation path "
            "natively. The tool/RAG (`documents`) branch is off the generation path "
            "and intentionally not covered."
        ),
        "source_repo": "mlx-community/c4ai-command-r7b-12-2024-4bit",
        "template_name": "default",
        "template": original,
        "cases": out_cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
