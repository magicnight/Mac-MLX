// ParametersInspector.swift
// macMLX
//
// Right-side Inspector panel for sampling + system prompt tuning.
// Attached to ChatView via SwiftUI's .inspector modifier.
// Spec: .claude/features/parameters.md (v0.2 MVP subset — issue #15).

import SwiftUI
import MacMLXCore

struct ParametersInspector: View {

    @Environment(AppState.self) private var appState

    var body: some View {
        @Bindable var params = appState.parameters
        Form {
            Section("Sampling") {
                parameterRow(
                    "Temperature",
                    help: "Higher = more creative, lower = more focused. 0.7 is a sensible default.",
                    value: params.parameters.temperature,
                    binding: Binding(
                        get: { params.parameters.temperature },
                        set: { params.parameters.temperature = $0; params.persist() }
                    ),
                    range: 0...2,
                    step: 0.05,
                    format: "%.2f"
                )

                parameterRow(
                    "Top P",
                    help: "Nucleus sampling: limit tokens to the most likely set summing to this probability.",
                    value: params.parameters.topP,
                    binding: Binding(
                        get: { params.parameters.topP },
                        set: { params.parameters.topP = $0; params.persist() }
                    ),
                    range: 0.05...1,
                    step: 0.05,
                    format: "%.2f"
                )

                HStack {
                    Text("Max Tokens")
                        .frame(width: 130, alignment: .leading)
                    // Direct numeric entry — users regularly jump from 2048
                    // to 16384 and the pre-v0.3.1 Stepper-only flow took
                    // ~112 clicks at step=128. TextField accepts any int,
                    // clamped to [128, 32768] on commit. Side Stepper
                    // preserves the "±128 nudge" affordance.
                    TextField(
                        "",
                        value: Binding(
                            get: { params.parameters.maxTokens },
                            set: { newValue in
                                let clamped = max(128, min(32768, newValue))
                                params.parameters.maxTokens = clamped
                                params.persist()
                            }
                        ),
                        format: .number
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 80)
                    .multilineTextAlignment(.trailing)

                    Stepper(
                        "",
                        value: Binding(
                            get: { params.parameters.maxTokens },
                            set: { params.parameters.maxTokens = $0; params.persist() }
                        ),
                        in: 128...32768,
                        step: 128
                    )
                    .labelsHidden()
                    .help("Maximum tokens to generate (128–32768). Type directly or ±128 with the stepper.")
                }
            }

            Section("System Prompt") {
                TextEditor(text: Binding(
                    get: { params.parameters.systemPrompt },
                    set: { params.parameters.systemPrompt = $0; params.persist() }
                ))
                .font(.body)
                .frame(minHeight: 80, idealHeight: 120)
            }

            Section {
                if let modelID = params.currentModelID {
                    Text("Current model: \(modelID)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else {
                    Text("No model loaded — changes apply once you load one.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }

                Button("Reset to Defaults") {
                    params.resetToDefaults()
                }
                .disabled(params.parameters == .default)
            } footer: {
                Text("Parameters persist per-model at ~/.mac-mlx/model-params/.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Parameters")
        // Reload stored values whenever the active model changes.
        .task(id: appState.coordinator.currentModel?.id) {
            await appState.parameters.loadForModel(appState.coordinator.currentModel?.id)
        }
    }

    // MARK: - Slider row helper

    private func parameterRow(
        _ label: String,
        help: String,
        value: Double,
        binding: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        format: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .frame(width: 130, alignment: .leading)
                Slider(value: binding, in: range, step: step)
                Text(String(format: format, value))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 45, alignment: .trailing)
            }
            .help(help)
        }
    }
}

#Preview {
    ParametersInspector()
        .environment(AppState())
        .frame(width: 320, height: 500)
}
