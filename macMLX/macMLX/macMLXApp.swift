// macMLXApp.swift
// macMLX
//
// App entry. Owns the AppState root, runs bootstrap on first appearance,
// and dispatches to MainWindowView (or OnboardingWindow once Task 4 lands).

import SwiftUI

@main
struct macMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(appState)
                .task {
                    await appState.bootstrap()
                    // Wire the menu bar manager once AppState is ready.
                    appDelegate.menuBarManager.setup(appState: appState)
                }
        }
        .windowResizability(.contentSize)
    }
}

/// Branches between Onboarding and MainWindow based on settings state.
private struct RootView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        // TODO(Task 4): wire OnboardingWindow when settings.onboardingComplete == false.
        // For now everyone lands on MainWindow until onboarding ships.
        MainWindowView()
    }
}
