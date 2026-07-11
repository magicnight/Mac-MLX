import Foundation

// Built-in chat-template override for `model_type: cohere2` (macMLX).
//
// WHY THIS EXISTS
// ---------------
// The real `mlx-community/c4ai-command-r7b-12-2024-4bit` checkpoint ships its
// `chat_template` as a LIST of three NAMED templates (`default` / `tool_use` /
// `rag`). The `default` template — the one transformers selects when neither
// `tools` nor `documents` is supplied — embeds a large tool/RAG branch behind
// `{%- if documents -%}` (with `document_turn` / `tool_call_id_to_int` /
// `format_tool_message` macros). swift-jinja 2.3.6 (the newest release) cannot
// PARSE that branch — compiling the full template throws
// "parser("Unexpected token type: closeExpression")" — so the whole template
// fails to compile even though the tool/RAG branch is never taken on the standard
// conversation path. That surfaces as `EngineError.modelLoadFailed` from the
// engine's lazy input-prep, blocking ALL end-to-end generation even though the
// Cohere2 architecture itself is parity-proven at 1e-4 (`Cohere2ModelParityTests`).
//
// `ChatTemplateOverride` registers this string for `cohere2` so it reaches the
// tokenizer (as `ChatTemplateArgument.literal`) INSTEAD of the checkpoint's own
// template, before any compilation.
//
// WHAT CHANGED VS THE ORIGINAL
// ----------------------------
// The `default` template's off-path tool/RAG branch (its `{% if documents %}`
// body) is DROPPED — replaced by an empty branch — while the `{%- else -%}`
// standard-conversation branch is kept BYTE-FOR-BYTE, including the outer
// `{% if documents %}…{%- else -%}…{% endif %}` skeleton so the whitespace-control
// behaviour is identical. Because transformers routes `documents`/`tools` to the
// SEPARATE `rag` / `tool_use` named templates, the `default` template is only ever
// rendered with `documents` falsy — so its `{% if documents %}` branch is dead on
// every real use, and dropping it changes nothing observable. Equivalence is
// proven ungated by `Cohere2ChatTemplateParityTests` (swift-jinja render of this
// override == Python jinja2 render of the ORIGINAL `default` template, for
// representative message sets) and end-to-end by the real-weights
// `Cohere2SmokeTests`.
//
// SCOPE / LIMITATION
// ------------------
// Tool-use and RAG conversations (which would need the `tool_use` / `rag` named
// templates) are out of scope for this override — macMLX's chat path renders
// plain multi-turn system/user/assistant conversations, which this reproduces
// exactly. A power user who needs the tool/RAG templates can drop a per-model
// `<model dir>/macmlx.chat_template.jinja` file, which takes precedence over this
// built-in (see `ChatTemplateOverride`).
//
// REMOVAL CONDITION
// -----------------
// Delete this override (and its `ChatTemplateOverride` registration) once
// swift-jinja can parse the full Command R7B `default` template — at which point
// the checkpoint's own template compiles directly and no rewrite is needed.

/// The built-in Cohere2 (Command R7B) chat template: the checkpoint's `default`
/// named template with its unparseable (and, for the default template, dead)
/// tool/RAG branch dropped and the standard-conversation branch kept verbatim.
/// See the file header for the full rationale, the exact diff, and the removal
/// condition.
enum Cohere2ChatTemplate {
    /// Upstream `default` template with the off-path `{% if documents %}` branch
    /// emptied; the `{%- else -%}` standard path is byte-for-byte upstream.
    static let template = #"""
{% if documents %}
{%- else -%}
{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}
    {%- set system_message = messages[0]['content'] %}{% elif false == true %}
    {%- set loop_messages = messages %}{% set system_message = '' %}
{%- else %}
    {%- set loop_messages = messages %}
    {%- set system_message = false %}
{%- endif %}
{%- if system_message != false -%}
    {{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}
{%- else -%}
    {{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif -%}
    {%- set content = message['content'] -%}
    {%- if message['role'] == 'user' -%}
        {{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>'  + content.strip() + '<|END_RESPONSE|><|END_OF_TURN_TOKEN|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt -%}
    {{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>' }}
{%- endif %}
{% endif %}
"""#
}
