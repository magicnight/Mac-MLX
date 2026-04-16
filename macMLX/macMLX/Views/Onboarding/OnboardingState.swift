// OnboardingState.swift
// macMLX
//
// @Observable step machine driving the 5-step first-launch wizard.
// Persists progress after each step so the user can quit and resume.

import Foundation
import MacMLXCore

@Observable
@MainActor
final class OnboardingState {

    // MARK: - Step enumeration

    enum Step: Int, CaseIterable {
        case welcome         = 0
        case modelDirectory  = 1
        case engineCheck     = 2
        case downloadModel   = 3
        case done            = 4

        var totalCount: Int { Step.allCases.count }
    }

    // MARK: - Observable state

    var currentStep: Step = .welcome
    var selectedModelDirectory: URL?
    var skipDownload: Bool = false
    var isComplete: Bool = false

    // MARK: - Private refs

    private let appState: AppState

    // MARK: - Init

    init(appState: AppState) {
        self.appState = appState
        // Restore partial state from settings
        let dir = appState.currentSettings.modelDirectory
        let home = FileManager.default.homeDirectoryForCurrentUser
        // Only pre-fill if it differs from the bare home default
        if dir != home.appending(path: "models") {
            selectedModelDirectory = dir
        }
    }

    // MARK: - Navigation

    func advance() {
        let next = currentStep.rawValue + 1
        if next < Step.allCases.count {
            currentStep = Step(rawValue: next)!
        }
    }

    func goBack() {
        let prev = currentStep.rawValue - 1
        if prev >= 0 {
            currentStep = Step(rawValue: prev)!
        }
    }

    // MARK: - Completion

    func complete() async {
        isComplete = true
        // Capture value before crossing into @Sendable closure (Swift 6 strict concurrency).
        let dir = selectedModelDirectory
        await appState.updateSettings { settings in
            settings.onboardingComplete = true
            if let dir {
                settings.modelDirectory = dir
            }
        }
    }

    func skip() async {
        // Save partial progress and mark complete so the wizard doesn't re-appear.
        await complete()
    }

    // MARK: - Step 2: directory persistence

    func confirmDirectory(_ url: URL) async {
        selectedModelDirectory = url
        await appState.updateSettings { settings in
            settings.modelDirectory = url
        }
    }
}
