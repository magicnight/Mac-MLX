import Foundation
import Jinja

extension MLXSwiftEngine {
    /// The chat template's `additionalContext`: the per-model template kwargs
    /// and, when the request has tools, those tools under `tools` as
    /// `Jinja.Value`s that keep the client's key order.
    ///
    /// `UserInput.tools` cannot carry the order. A `ToolSpec` is a Swift
    /// dictionary, and swift-transformers sorts a dictionary's keys when it
    /// builds the template context, so `tool | tojson` printed
    /// `{"function": {"description": …, "name": …}, "type": "function"}` where
    /// Python prints the keys as the client sent them, `type` first. The
    /// checkpoints were trained on the latter. swift-transformers applies
    /// `additionalContext` after it sets `tools` and passes a `Jinja.Value`
    /// through untouched, so the entry here is the list the template reads.
    /// `UserInput.tools` still goes along: it selects a `tool_use` named
    /// template and is what a processor looks at.
    ///
    /// The list has the same elements as `toolSpecs(from:)` (objects only). A
    /// `tools` template kwarg is the user's own and is left alone.
    static func templateContext(kwargs: [String: JSONValue]?, tools: [JSONValue]?) -> [String: any Sendable]? {
        var context = kwargs?.mapValues { $0.toTemplateValue() }
        guard let tools, kwargs?["tools"] == nil else { return context }
        let ordered = tools.filter { if case .object = $0 { true } else { false } }.map { $0.toJinjaValue() }
        context = context ?? [:]
        context?["tools"] = Jinja.Value.array(ordered)
        return context
    }
}
