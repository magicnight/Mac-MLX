import Foundation
import Jinja
import Testing

/// Three places where swift-jinja 2.x rendered a chat template differently from
/// Python's Jinja2, each of which changed a real prompt: Gemma 4's tool
/// declarations carried stray spaces, Seed-OSS and Qwen 3.5 replayed a boolean
/// tool argument as `true` where Python renders `True`, and the
/// rerankers' templates kept a trailing newline Python drops. swift-jinja 3
/// follows Jinja2 on all three. The expected strings are what
/// `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`, the
/// environment transformers renders chat templates in, returns for the same
/// template and context (Jinja2 3.1.6). Pure; no model, no Metal.
@Suite("Jinja Python parity")
struct JinjaPythonParityTests {

    /// Same options swift-transformers uses when it compiles a chat template.
    private func render(_ source: String, _ context: [String: Value] = [:]) throws -> String {
        try Template(source, with: .init(lstripBlocks: true, trimBlocks: true)).render(context)
    }

    /// Gemma 4's tool declaration opens a brace, breaks the line, and continues
    /// with a tag that strips to its left: `parameters:{`, newline, indent,
    /// `{%- if … -%}`. swift-jinja 2.5.1 left one space after the brace
    /// (`p:{ q}`), and the same after `properties:{ {{- … -}} }`.
    @Test
    func aLeftStrippingTagRemovesTheWhitespaceAfterALiteralBrace() throws {
        #expect(
            try render("p:{\n        {%- if a -%}\n            q\n        {%- endif -%}\n}", ["a": true])
                == "p:{q}")
        #expect(
            try render("p:{\n    {%- if a -%}\n        q:{ {{- 'z' -}} },\n    {%- endif -%}\n}", ["a": true])
                == "p:{q:{z},}")
    }

    @Test
    func booleansAndNonePrintAsPythonDoes() throws {
        #expect(
            try render("<parameter={{ name }}>{{ value }}</parameter>", ["name": "metric", "value": true])
                == "<parameter=metric>True</parameter>")
        #expect(
            try render("{{ none_value }}|{{ flags }}", ["none_value": .null, "flags": [true, false, "a"]])
                == "None|[True, False, 'a']")
    }

    /// Not parity: Python has no such limit here. swift-jinja 3 stops a macro
    /// that calls itself at 16 levels, and Gemma 4's and MiniCPM3's templates
    /// walk a tool's parameter schema with one. Pinned so that a change to the
    /// limit, in either direction, is noticed.
    @Test
    func aRecursiveMacroRendersTenLevelsAndStopsPastSixteen() throws {
        let source =
            "{%- macro walk(node) -%}"
            + "{{ node['name'] }}{%- if node['child'] -%}>{{ walk(node['child']) }}{%- endif -%}"
            + "{%- endmacro -%}{{ walk(root) }}"
        func chain(_ depth: Int) -> Value {
            var node: Value = ["name": .string("n\(depth)")]
            for level in stride(from: depth - 1, through: 1, by: -1) {
                node = ["name": .string("n\(level)"), "child": node]
            }
            return node
        }
        #expect(try render(source, ["root": chain(10)]) == (1...10).map { "n\($0)" }.joined(separator: ">"))
        #expect(throws: (any Error).self) { try render(source, ["root": chain(20)]) }
    }

    @Test
    func oneTrailingNewlineIsDropped() throws {
        #expect(try render("<think>\n\n</think>\n\n\n") == "<think>\n\n</think>\n\n")
    }
}
