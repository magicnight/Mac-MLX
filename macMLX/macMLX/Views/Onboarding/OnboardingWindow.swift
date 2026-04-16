// OnboardingWindow.swift
// macMLX
//
// Container view that hosts the 5-step onboarding wizard.
// Rendered as a SwiftUI sheet on the main window (safer than a bare
// NSWindow in a SwiftUI lifecycle). The sheet is shown from RootView
// when onboardingComplete == false.

import SwiftUI
import MacMLXCore

struct OnboardingWindow: View {

    @Environment(AppState.self) private var appState
    @State private var onboardingState: OnboardingState?

    // Called by parent when wizard completes or is skipped.
    let onComplete: () -> Void

    var body: some View {
        Group {
            if let state = onboardingState {
                OnboardingContent(state: state, onComplete: onComplete)
            } else {
                ProgressView("Loading…")
                    .frame(width: 480, height: 400)
            }
        }
        .task {
            onboardingState = OnboardingState(appState: appState)
        }
    }
}

// MARK: - OnboardingContent

private struct OnboardingContent: View {

    @Bindable var state: OnboardingState
    @Environment(AppState.self) private var appState

    let onComplete: () -> Void

    var body: some View {
        Group {
            switch state.currentStep {
            case .welcome:
                WelcomeStep(
                    onGetStarted: { state.advance() },
                    onSkip: {
                        Task {
                            await state.skip()
                            onComplete()
                        }
                    }
                )

            case .modelDirectory:
                ModelDirectoryStep(state: state)

            case .engineCheck:
                EngineCheckStep(state: state)

            case .downloadModel:
                // Skip step 4 if the user already has MLX models
                let modelDir = appState.currentSettings.modelDirectory
                let hasModels = (try? FileManager.default
                    .contentsOfDirectory(at: modelDir,
                                        includingPropertiesForKeys: nil)
                    .filter { url in
                        let items = (try? FileManager.default
                            .contentsOfDirectory(at: url,
                                                includingPropertiesForKeys: nil)
                            .map { $0.lastPathComponent }) ?? []
                        return ModelFormat.detect(in: items) == .mlx
                    }
                    .isEmpty == false) ?? false

                if hasModels {
                    // Auto-advance to Done
                    Color.clear
                        .frame(width: 1, height: 1)
                        .task { state.advance() }
                } else {
                    DownloadModelStep(state: state)
                        .environment(appState)
                }

            case .done:
                DoneStep {
                    await state.complete()
                    onComplete()
                }
            }
        }
        .frame(width: 540, height: 520)
        .animation(.easeInOut(duration: 0.2), value: state.currentStep)
    }
}

#Preview {
    OnboardingWindow(onComplete: {})
        .environment(AppState())
}
