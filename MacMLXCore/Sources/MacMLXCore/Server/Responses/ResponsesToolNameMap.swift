// Copyright © 2026 macMLX. English comments only.

/// The names a Responses request's tools are offered to the model under, and
/// the way back from a name the model calls to the tool the client declared.
///
/// A Responses client can group function tools in a `namespace` (Codex's
/// `multi_agent_v1`) and expects a call to come back as the BARE tool name
/// plus a `namespace` field. The chat template has one flat list of function
/// names, so namespaced tools are flattened into it. A bare name keeps its
/// name; one that collides — the same name in two namespaces, or in a
/// namespace and at the top level — is offered as `<namespace>__<name>`
/// instead, and only the namespaced ones are renamed (a top-level tool keeps
/// its name). A `custom` (freeform) tool is offered as a function too, and its
/// kind is kept so the call comes back as a `custom_tool_call`.
struct ResponsesToolNameMap: Equatable, Sendable {

    enum Kind: Equatable, Sendable {
        case function
        case custom
    }

    /// One declared tool: where it lives and what kind of call it expects.
    struct Entry: Equatable, Sendable {
        let namespace: String?
        let name: String
        let kind: Kind
    }

    /// Exposed name → declared tool.
    private let entries: [String: Entry]

    /// Builds the map from every declared tool, in declaration order. Exposed
    /// names are computed over the whole set, so a collision is detected no
    /// matter which tool comes first.
    init(_ declared: [Entry]) {
        var bareCounts: [String: Int] = [:]
        for entry in declared {
            bareCounts[entry.name, default: 0] += 1
        }
        var entries: [String: Entry] = [:]
        for entry in declared {
            entries[Self.exposedName(for: entry, bareCounts: bareCounts)] = entry
        }
        self.entries = entries
    }

    /// The map of a request with no tools.
    static let empty = ResponsesToolNameMap([])

    /// The declared tool the model's call `exposedName` resolves to, or nil
    /// for a name the request never declared.
    func entry(forExposedName exposedName: String) -> Entry? {
        entries[exposedName]
    }

    /// The name the model knows a declared tool by — used to replay a call
    /// from the client's history (bare name plus `namespace`) into the name
    /// the model was offered. Nil when no declared tool matches.
    func exposedName(namespace: String?, name: String) -> String? {
        entries.first { $0.value.namespace == namespace && $0.value.name == name }?.key
    }

    private static func exposedName(for entry: Entry, bareCounts: [String: Int]) -> String {
        guard let namespace = entry.namespace, (bareCounts[entry.name] ?? 0) > 1 else {
            return entry.name
        }
        return "\(namespace)__\(entry.name)"
    }
}
