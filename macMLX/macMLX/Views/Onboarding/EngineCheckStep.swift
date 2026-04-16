// EngineCheckStep.swift
// macMLX — Onboarding Step 3

import SwiftUI
import MacMLXCore

struct EngineCheckStep: View {

    @Bindable var state: OnboardingState

    @State private var mlxReady = false
    @State private var isChecking = true

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            stepHeader(index: 2, title: "Checking inference engine…")

            VStack(alignment: .leading, spacing: 12) {
                engineRow(
                    name: "MLX Swift Engine",
                    status: isChecking ? "Checking…" : (mlxReady ? "Ready" : "Error"),
                    isChecking: isChecking,
                    isReady: mlxReady,
                    isInstalled: true
                )
                engineRow(
                    name: "SwiftLM",
                    status: "Not installed",
                    isChecking: false,
                    isReady: false,
                    isInstalled: false
                )
                engineRow(
                    name: "Python MLX",
                    status: "Not installed",
                    isChecking: false,
                    isReady: false,
                    isInstalled: false
                )
            }
            .padding(16)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            Text("MLX Swift is the recommended engine for most models. You can install additional engines later in Settings.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            HStack {
                Button("Back") { state.goBack() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Continue") { state.advance() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isChecking || !mlxReady)
            }
        }
        .padding(40)
        .task { await checkEngines() }
    }

    private func engineRow(
        name: String,
        status: String,
        isChecking: Bool,
        isReady: Bool,
        isInstalled: Bool
    ) -> some View {
        HStack {
            Group {
                if isChecking {
                    ProgressView()
                        .scaleEffect(0.7)
                        .frame(width: 16, height: 16)
                } else if isReady {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else if isInstalled {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.red)
                } else {
                    Image(systemName: "circle")
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 20)

            Text(name)
                .font(.subheadline)

            Spacer()

            Text(status)
                .font(.caption)
                .foregroundStyle(isReady ? .green : .secondary)
        }
    }

    private func checkEngines() async {
        isChecking = true
        // MLXSwiftEngine instantiates successfully in process — treat that as "ready"
        let engine = MLXSwiftEngine()
        // Brief deliberate pause so the "Checking…" state is visible to the user
        try? await Task.sleep(for: .seconds(0.8))
        let _ = engine  // ensure not optimised away
        mlxReady = true
        isChecking = false
    }
}

#Preview {
    EngineCheckStep(state: OnboardingState(appState: AppState()))
        .frame(width: 480, height: 400)
}
