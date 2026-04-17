// EnginePickerSection.swift
// macMLX
//
// Form section for choosing the active inference engine.
// v0.1: only .mlxSwift is functional; swiftLM and pythonMLX are detection-only
// with clear "Coming in v0.2" messaging so users don't mistake them for broken
// installs.

import SwiftUI
import MacMLXCore

struct EnginePickerSection: View {

    @Binding var selectedEngine: EngineID
    let onEngineChange: (EngineID) -> Void

    var body: some View {
        Section("Inference Engine") {
            ForEach(EngineID.allCases, id: \.self) { engine in
                engineRow(engine)
            }
        }
    }

    private func engineRow(_ engine: EngineID) -> some View {
        let state = availability(engine)
        return HStack(alignment: .top, spacing: 12) {
            radioIcon(for: engine, state: state)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 4) {
                // Title row: name + Recommended / v0.2 badge
                HStack(spacing: 6) {
                    Text(engine.displayName)
                        .font(.subheadline)
                        .foregroundStyle(state == .available ? .primary : .secondary)
                    badge(for: engine, state: state)
                }

                // One-liner description
                Text(engine.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // Unavailable engines get a friendly explanation under the description.
                if state != .available {
                    Text(state.unavailableHint)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 1)
                }
            }

            Spacer()
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if state == .available {
                selectedEngine = engine
                onEngineChange(engine)
            }
        }
        // Previous 0.6 felt "看不见" — bump to 0.8 so copy stays readable
        // even on deferred rows; the distinction is now carried by the badge.
        .opacity(state == .available ? 1 : 0.8)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(state == .available ? .isButton : [])
    }

    // MARK: - Subviews

    private func radioIcon(for engine: EngineID, state: Availability) -> some View {
        let isSelected = selectedEngine == engine
        let iconName = isSelected ? "largecircle.fill.circle" : "circle"
        return Image(systemName: iconName)
            .foregroundStyle(
                isSelected ? Color.accentColor
                           : (state == .available ? Color.secondary : Color.secondary.opacity(0.6))
            )
    }

    @ViewBuilder
    private func badge(for engine: EngineID, state: Availability) -> some View {
        switch (engine, state) {
        case (.mlxSwift, .available):
            pill("Recommended", fg: .white, bg: .green)
        case (_, .comingInV02):
            pill("v0.2", fg: .white, bg: .orange)
        default:
            EmptyView()
        }
    }

    private func pill(_ text: String, fg: Color, bg: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(fg)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(bg, in: Capsule())
    }

    // MARK: - Availability model

    private enum Availability {
        case available           // functional in v0.1
        case comingInV02         // detected but not yet wired

        var unavailableHint: String {
            switch self {
            case .available:    return ""
            case .comingInV02:  return "Not yet wired. Coming in v0.2."
            }
        }
    }

    private func availability(_ engine: EngineID) -> Availability {
        // v0.1 scope: only MLX Swift is functional. swiftLM / pythonMLX
        // are detection-only per .claude/features/inference-engines.md.
        engine == .mlxSwift ? .available : .comingInV02
    }
}

// MARK: - EngineID display helpers

extension EngineID {
    var displayName: String {
        switch self {
        case .mlxSwift:  return "MLX Swift Engine"
        case .swiftLM:   return "SwiftLM"
        case .pythonMLX: return "Python MLX"
        }
    }

    var description: String {
        switch self {
        case .mlxSwift:
            return "In-process · Apple mlx-swift-lm · Default for most models"
        case .swiftLM:
            return "External binary · Built for 100B+ MoE models with SSD streaming"
        case .pythonMLX:
            return "uv-managed Python subprocess · Widest model compatibility"
        }
    }
}

#Preview {
    Form {
        EnginePickerSection(
            selectedEngine: .constant(.mlxSwift),
            onEngineChange: { _ in }
        )
    }
    .formStyle(.grouped)
    .frame(width: 500)
}
