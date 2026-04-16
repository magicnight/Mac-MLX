// EnginePickerSection.swift
// macMLX
//
// Form section for choosing the active inference engine.
// v0.1: only .mlxSwift is functional; swiftLM and pythonMLX are detection-only.

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
        HStack {
            // Radio-style selection
            Image(systemName: selectedEngine == engine
                  ? "largecircle.fill.circle"
                  : "circle")
                .foregroundStyle(selectedEngine == engine ? Color.accentColor : Color.secondary)
                .onTapGesture {
                    if isAvailable(engine) {
                        selectedEngine = engine
                        onEngineChange(engine)
                    }
                }

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(engine.displayName)
                        .font(.subheadline)
                        .foregroundStyle(isAvailable(engine) ? .primary : .secondary)

                    if engine == .mlxSwift {
                        Text("Recommended")
                            .font(.caption2)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.green, in: Capsule())
                    }

                    if !isAvailable(engine) {
                        Text("Not installed")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.15), in: Capsule())
                    }
                }

                Text(engine.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if isAvailable(engine) {
                selectedEngine = engine
                onEngineChange(engine)
            }
        }
        .opacity(isAvailable(engine) ? 1 : 0.6)
    }

    private func isAvailable(_ engine: EngineID) -> Bool {
        // v0.1: only mlxSwift is functional
        engine == .mlxSwift
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
            return "In-process · Apple official mlx-swift-lm · Default"
        case .swiftLM:
            return "External binary · Best for 100B+ MoE models"
        case .pythonMLX:
            return "uv-managed Python subprocess · Maximum model compatibility"
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
