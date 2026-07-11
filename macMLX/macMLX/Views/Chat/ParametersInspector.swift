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

            Section("LoRA Adapter") {
                HStack {
                    Picker(
                        "Adapter",
                        selection: Binding<String>(
                            get: { params.parameters.adapterName ?? "" },
                            set: { newValue in
                                let trimmed = newValue.isEmpty ? nil : newValue
                                params.parameters.adapterName = trimmed
                                params.persist()
                            }
                        )
                    ) {
                        Text("None").tag("")
                        ForEach(appState.availableAdapters, id: \.name) { adapter in
                            Text(adapter.name).tag(adapter.name)
                        }
                    }
                    .pickerStyle(.menu)
                    .help("Apply a LoRA adapter on top of the loaded base model. Drop PEFT-format adapters into ~/.mac-mlx/adapters/<name>/. The adapter applies on the next model load.")

                    Button {
                        Task { await appState.refreshAdapters() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .help("Re-scan ~/.mac-mlx/adapters/")
                }
                if appState.availableAdapters.isEmpty {
                    Text("No adapters in ~/.mac-mlx/adapters/. Drop a PEFT or mlx-format LoRA folder there and click Refresh.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let name = params.parameters.adapterName,
                   !name.isEmpty,
                   !appState.availableAdapters.contains(where: { $0.name == name }) {
                    Text("Configured adapter '\(name)' is not present — load will skip the adapter.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }

            Section("Speculative Decoding") {
                Picker(
                    "Draft Model",
                    selection: Binding<String>(
                        get: { params.parameters.draftModelID ?? "" },
                        set: { newValue in
                            let trimmed = newValue.isEmpty ? nil : newValue
                            params.parameters.draftModelID = trimmed
                            // Seed the token count so the persisted value
                            // matches what the "Draft Tokens" slider below
                            // already displays (it falls back to 2 when
                            // nil) — without this, the UI shows "2" but a
                            // freshly-picked draft model silently sends
                            // `numDraftTokens: nil` until the user touches
                            // the slider themselves.
                            if trimmed != nil, params.parameters.numDraftTokens == nil {
                                params.parameters.numDraftTokens = 2
                            }
                            params.persist()
                        }
                    )
                ) {
                    Text("None (disabled)").tag("")
                    ForEach(draftCandidates) { candidate in
                        Text(candidate.displayName).tag(candidate.id)
                    }
                }
                .pickerStyle(.menu)
                .help("Speculate with a smaller, faster model to accelerate generation. Leave as None to disable.")

                if let staleDraftModelID {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text("Draft model '\(staleDraftModelID)' unavailable")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("Reset to None") {
                            params.parameters.draftModelID = nil
                            params.persist()
                        }
                        .font(.caption)
                        .buttonStyle(.borderless)
                    }
                }

                if params.parameters.draftModelID != nil {
                    parameterRow(
                        "Draft Tokens",
                        help: "How many tokens the draft model proposes per speculative round (1–8). Higher can speed things up more but wastes more work whenever a proposal is rejected.",
                        value: Double(params.parameters.numDraftTokens ?? 2),
                        binding: Binding(
                            get: { Double(params.parameters.numDraftTokens ?? 2) },
                            set: { newValue in
                                params.parameters.numDraftTokens = Int(newValue.rounded())
                                params.persist()
                            }
                        ),
                        range: 1...8,
                        step: 1,
                        format: "%.0f"
                    )
                }

                if draftCandidates.isEmpty {
                    Text("No other text models in your library — download or add one to use as a draft model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Text("Not every model supports speculative decoding — hybrid/linear-attention architectures (e.g. Qwen3.5) silently fall back to plain generation instead of erroring. You'll just see no acceptance rate on that turn.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
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

    // MARK: - Speculative decoding (Track F GUI over the D1 engine plumbing)

    /// Candidate draft models. Filtering logic lives in
    /// `LocalModel.draftCandidates(from:excluding:)` (Core-side, unit
    /// tested) — this just supplies the current library + target model id.
    private var draftCandidates: [LocalModel] {
        LocalModel.draftCandidates(
            from: appState.modelLibrary.localModels,
            excluding: appState.parameters.currentModelID
        )
    }

    /// The persisted `draftModelID` when it no longer names a valid
    /// candidate (deleted from the library, or now the loaded target
    /// itself) — `nil` otherwise. The Picker's binding falls back to `""`
    /// in that case (SwiftUI shows no visible selection) while
    /// `draftModelID` stays set on disk, so generation keeps trying — and
    /// failing — to load it every round.
    private var staleDraftModelID: String? {
        guard let id = appState.parameters.parameters.draftModelID, !id.isEmpty,
              !draftCandidates.contains(where: { $0.id == id })
        else { return nil }
        return id
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
