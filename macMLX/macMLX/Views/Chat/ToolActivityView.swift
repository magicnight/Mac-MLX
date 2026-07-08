// ToolActivityView.swift
// macMLX
//
// Native rendering for one MCP tool-loop row (v0.6 wave 2): the assistant's
// tool call, or the tool's result. A compact monospaced card with an SF Symbol
// header; long argument/result bodies collapse behind a disclosure so a chatty
// tool doesn't flood the transcript. Auto-run — no approval controls this wave.

import SwiftUI

struct ToolActivityView: View {

    let activity: UIChatMessage.ToolActivity
    /// Body text: the result text for `.result`, ignored for `.call` (which
    /// renders its `argumentsJSON` instead).
    let content: String

    @State private var expanded = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            card
            Spacer(minLength: 60)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
    }

    // MARK: - Card

    private var card: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            if !bodyText.isEmpty {
                if isLong {
                    DisclosureGroup(isExpanded: $expanded) {
                        bodyView
                    } label: {
                        Text(expanded ? "Hide details" : "Show details")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    bodyView
                }
            }
        }
        .padding(10)
        .frame(maxWidth: 460, alignment: .leading)
        .background(background, in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(borderColor, lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: iconName)
                .font(.caption)
                .foregroundStyle(iconColor)
            Text(title)
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.semibold)
                .foregroundStyle(.primary)
            if let subtitle {
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
    }

    private var bodyView: some View {
        Text(bodyText)
            .font(.system(.caption2, design: .monospaced))
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Derived presentation

    private var bodyText: String {
        switch activity {
        case .call(_, _, let argumentsJSON):
            // Hide the empty "{}" object — a no-argument call needs no body.
            return argumentsJSON == "{}" ? "" : argumentsJSON
        case .result:
            return content
        }
    }

    /// Long bodies collapse behind a disclosure so verbose tool output doesn't
    /// dominate the transcript.
    private var isLong: Bool {
        bodyText.count > 220 || bodyText.split(separator: "\n", omittingEmptySubsequences: false).count > 6
    }

    private var iconName: String {
        switch activity {
        case .call:
            return "wrench.and.screwdriver"
        case .result(let isError):
            return isError ? "exclamationmark.triangle" : "terminal"
        }
    }

    private var iconColor: Color {
        switch activity {
        case .call:
            return .accentColor
        case .result(let isError):
            return isError ? .red : .secondary
        }
    }

    private var title: String {
        switch activity {
        case .call(let name, _, _):
            return name
        case .result(let isError):
            return isError ? "Tool error" : "Tool result"
        }
    }

    private var subtitle: String? {
        switch activity {
        case .call(_, let server, _):
            return "→ \(server)"
        case .result:
            return nil
        }
    }

    private var isError: Bool {
        if case .result(let e) = activity { return e }
        return false
    }

    private var background: Color {
        isError ? Color.red.opacity(0.08) : Color(.tertiarySystemFill)
    }

    private var borderColor: Color {
        isError ? Color.red.opacity(0.35) : Color.secondary.opacity(0.25)
    }
}

#Preview {
    VStack(alignment: .leading) {
        ToolActivityView(
            activity: .call(
                name: "get_weather",
                server: "weather",
                argumentsJSON: "{\n  \"city\" : \"Paris\",\n  \"units\" : \"metric\"\n}"),
            content: "")
        ToolActivityView(
            activity: .result(isError: false),
            content: "It is currently 18°C and sunny in Paris.")
        ToolActivityView(
            activity: .result(isError: true),
            content: "Error: tool 'get_weather' failed: connection refused")
    }
    .frame(width: 520)
    .padding()
}
