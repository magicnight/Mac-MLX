import Foundation

// Built-in chat-template override for `model_type: seed_oss` (macMLX).
//
// WHY THIS EXISTS
// ---------------
// The real `mlx-community/Seed-OSS-36B-Instruct-4bit` checkpoint ships a
// `chat_template.jinja` that builds its thinking-budget reflection-interval
// table as a Jinja object literal with INTEGER keys:
//
//     {%- set budget_reflections_v05 = {0: 0, 512: 128, 1024: 256, ...} -%}
//
// swift-jinja 2.3.6 (the newest release; no upstream fix exists) parses only
// string/identifier object keys, so compiling the stock template throws
// "Parser error: Expected string literal or identifier for object key. Got
// number instead". That surfaces as `EngineError.modelLoadFailed` from the
// engine's lazy input-prep, blocking ALL end-to-end generation even though the
// Seed-OSS architecture itself is parity-proven at 1e-4
// (`SeedOss{Attention,MLP,Model}ParityTests`).
//
// `ChatTemplateOverride` registers this string for `seed_oss` so it reaches the
// tokenizer (as `ChatTemplateArgument.literal`) INSTEAD of the checkpoint's own
// template, before any compilation.
//
// WHAT CHANGED VS THE ORIGINAL
// ----------------------------
// EXACTLY ONE construct. The integer-keyed dict was consumed only as a
// sorted-ascending threshold search (`dictsort` then the first tier whose key
// is >= `thinking_budget` wins) plus a `[16384]` fallback for budgets past the
// top tier. That is rewritten as an equivalent `if/elif` ladder over the same
// tier boundaries, emitting an identical `ns.interval` for every
// `thinking_budget`. Everything else — special-token setup, system handling,
// tool-def emission, thinking-budget system blocks, the message loop, and the
// generation prompt — is byte-for-byte the upstream template. Equivalence is
// proven ungated by `SeedOssChatTemplateParityTests` (swift-jinja render of this
// override == Python jinja2 render of the ORIGINAL, for representative message
// sets) and end-to-end by the real-weights `SeedOssSmokeTests`.
//
// REMOVAL CONDITION
// -----------------
// Delete this override (and its `ChatTemplateOverride` registration) once
// swift-jinja gains integer object-key support — at which point the checkpoint's
// own template parses directly and no rewrite is needed.
//
// A per-model user file `<model dir>/macmlx.chat_template.jinja` still takes
// precedence over this built-in (see `ChatTemplateOverride`).

/// The built-in Seed-OSS chat template, semantically identical to the upstream
/// `chat_template.jinja` but with the integer-keyed budget dict rewritten as an
/// `if/elif` ladder swift-jinja can parse. See the file header for the full
/// rationale, the exact diff, and the removal condition.
enum SeedOssChatTemplate {
    /// Verbatim upstream template with the single unparseable dict rewritten.
    static let template = #"""
{# ----------‑‑‑ special token variables ‑‑‑---------- #}
{%- set bos_token              = '<seed:bos>'               -%}
{%- set eos_token              = '<seed:eos>'               -%}
{%- set pad_token              = '<seed:pad>'               -%}
{%- set toolcall_begin_token   = '<seed:tool_call>'         -%}
{%- set toolcall_end_token     = '</seed:tool_call>'        -%}
{%- set think_begin_token      = '<seed:think>'             -%}
{%- set think_end_token        = '</seed:think>'            -%}
{%- set budget_begin_token     = '<seed:cot_budget_reflect>'-%}
{%- set budget_end_token       = '</seed:cot_budget_reflect>'-%}
{# -------------- reflection-interval lookup -------------- #}
{%- if not thinking_budget is defined %}
{%- set thinking_budget = -1 -%}
{%- endif -%}
{# macMLX override: the upstream template built `budget_reflections_v05` as a
   Jinja object literal with INTEGER keys ({0: 0, 512: 128, ...}); swift-jinja
   2.3.6 cannot parse numeric object keys. That dict was consumed ONLY as a
   sorted-ascending threshold search — `dictsort` then the FIRST tier whose key
   is >= thinking_budget wins — with a `[16384]` fallback for budgets past the
   top tier. The if/elif ladder below is that exact search unrolled: it emits an
   identical `ns.interval` for every thinking_budget, so the rendered prompt is
   byte-for-byte unchanged. Remove this rewrite once swift-jinja parses integer
   object keys. #}
{%- set ns = namespace(interval = None) -%}
{%- if thinking_budget <= 0 -%}
    {%- set ns.interval = 0 -%}
{%- elif thinking_budget <= 512 -%}
    {%- set ns.interval = 128 -%}
{%- elif thinking_budget <= 1024 -%}
    {%- set ns.interval = 256 -%}
{%- elif thinking_budget <= 2048 -%}
    {%- set ns.interval = 512 -%}
{%- elif thinking_budget <= 4096 -%}
    {%- set ns.interval = 512 -%}
{%- elif thinking_budget <= 8192 -%}
    {%- set ns.interval = 1024 -%}
{%- elif thinking_budget <= 16384 -%}
    {%- set ns.interval = 1024 -%}
{%- else -%}
    {%- set ns.interval = 1024 -%}
{%- endif -%}
{# ---------- 预处理 system 消息 ---------- #}
{%- if messages[0]["role"] == "system" %}
{%- set system_message = messages[0]["content"] %}
{%- set loop_messages = messages[1:] %}
{%- else %}
{%- set loop_messages = messages %}
{%- endif %}
{# ---------- 确保 tools 存在 ---------- #}
{%- if not tools is defined or tools is none %}
{%- set tools = [] %}
{%- endif %}
{# tools2doc.jinja #}
{%- macro py_type(t) -%}
    {%- if t == "string" -%}str
    {%- elif t in ("number", "integer") -%}int
    {%- elif t == "boolean" -%}bool
    {%- elif t == "array" -%}list
    {%- else -%}Any{%- endif -%}
{%- endmacro -%}
{# ---------- 输出 system 块 ---------- #}
{%- if system_message is defined %}
{{ bos_token + "system\n" + system_message }}
{%- else %}
{%- if tools is iterable and tools | length > 0 %}
{{ bos_token + "system\nYou are Doubao, a helpful AI assistant. You may call one or more functions to assist with the user query." }}
{%- endif %}
{%- endif %}
{%- if use_json_tooldef is defined and use_json_tooldef %}

{{"Tool List:\nYou are authorized to use the following tools (described in JSON Schema format). Before performing any task, you must decide how to call them based on the descriptions and parameters of these tools."}}
{{ tools | tojson(ensure_ascii=False) }}
{%- else %}
{%- for item in tools if item.type == "function" %}


Function:
def {{ item.function.name }}(
{%- for name, spec in item.function.parameters.properties.items() %}
        {{- name }}: {{ py_type(spec.type) }}{% if not loop.last %},{% endif %}
{%- endfor %}):
    """
    {{ item.function.description | trim }}

    {# ---------- Args ---------- #}
    {%- if item.function.parameters.properties %}
    Args:
    {%- for name, spec in item.function.parameters.properties.items() %}

    - {{ name }} ({{ py_type(spec.type) }})
      {%- if name in item.function.parameters.required %} [必填]{% else %} [选填]{% endif %}:
      {{- " " ~ (spec.description or "") }}
    {%- endfor %}
    {%- endif %}

    {# ---------- Returns ---------- #}
    {%- if item.function.returns is defined
          and item.function.returns.properties is defined
          and item.function.returns.properties %}
    Returns:
    {%- for name, spec in item.function.returns.properties.items() %}

    - {{ name }} ({{ py_type(spec.type) }}):
      {{- " " ~ (spec.description or "") }}
    {%- endfor %}
    {%- endif %}

    """
{%- endfor %}
{%- endif %}
{%- if tools is iterable and tools | length > 0 %}

{{"工具调用请遵循如下格式:\n<seed:tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>value_1</parameter>\n<parameter=example_parameter_2>This is the value for the second parameter\nthat can span\nmultiple lines</parameter>\n</function>\n</seed:tool_call>\n"}}
{%- endif %}
{# 结束 system 块行尾 #}
{%- if system_message is defined or tools is iterable and tools | length > 0 %}
{{ eos_token }}
{%- endif %}
{# ---------- Thinking Budget ---------- #}
{%- if thinking_budget is defined %}
{%- if thinking_budget == 0 %}
{{ bos_token+"system" }}
{{ "You are an intelligent assistant that can answer questions in one step without the need for reasoning and thinking, that is, your thinking budget is 0. Next, please skip the thinking process and directly start answering the user's questions." }}
{{ eos_token }}
{%- elif not thinking_budget == -1 %}
{{ bos_token+"system" }}
{{ "You are an intelligent assistant with reflective ability. In the process of thinking and reasoning, you need to strictly follow the thinking budget, which is "}}{{thinking_budget}}{{". That is, you need to complete your thinking within "}}{{thinking_budget}}{{" tokens and start answering the user's questions. You will reflect on your thinking process every "}}{{ns.interval}}{{" tokens, stating how many tokens have been used and how many are left."}}
{{ eos_token }}
{%- endif %}
{%- endif %}
{# ---------- 逐条写出历史消息 ---------- #}
{%- for message in loop_messages %}
{%- if message.role == "assistant"
  and message.tool_calls is defined
  and message.tool_calls is iterable
  and message.tool_calls | length > 0 %}
{{ bos_token + message.role }}
{%- if message.reasoning_content is defined and message.reasoning_content is string and message.reasoning_content | trim | length > 0 %}
{{ "\n" + think_begin_token + message.reasoning_content | trim + think_end_token }}
{%- endif %}
{%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}
{{ "\n" + message.content | trim + "\n" }}
{%- endif %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function is defined %}{% set tool_call = tool_call.function %}{% endif %}
{{ "\n" + toolcall_begin_token + "\n<function=" + tool_call.name + ">\n" }}
{%- if tool_call.arguments is defined %}
{%- for arg_name, arg_value in tool_call.arguments | items %}
{{ "<parameter=" + arg_name + ">" }}
{%- set arg_value = arg_value if arg_value is string else arg_value | string %}
{{ arg_value+"</parameter>\n" }}
{%- endfor %}
{%- endif %}
{{ "</function>\n" + toolcall_end_token }}
{%- endfor %}
{{ eos_token }}
{%- elif message.role in ["user", "system"] %}
{{ bos_token + message.role + "\n" + message.content + eos_token }}
{%- elif message.role == "assistant" %}
{{ bos_token + message.role }}
{%- if message.reasoning_content is defined and message.reasoning_content is string and message.reasoning_content | trim | length > 0 %}
{{ "\n" + think_begin_token + message.reasoning_content | trim + think_end_token }}
{%- endif %}
{%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}
{{ "\n" + message.content | trim + eos_token }}
{%- endif %}
{# 包括 tool 角色，在这个逻辑 #}
{%- else %}
{{ bos_token + message.role + "\n" + message.content + eos_token }}
{%- endif %}
{%- endfor %}
{# ---------- 控制模型开始续写 ---------- #}
{%- if add_generation_prompt %}
{{ bos_token+"assistant\n" }}
{%- if thinking_budget == 0 %}
{{ think_begin_token+budget_begin_token }}
{%- endif %}
{%- endif %}
"""#
}
