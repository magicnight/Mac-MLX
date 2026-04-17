// macMLXApp.swift
// macMLX
//
// App entry. Owns the AppState root, runs bootstrap on first appearance,
// and dispatches to MainWindowView (or OnboardingWindow once Task 4 lands).

import SwiftUI
import MacMLXCore

@main
struct macMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @State private var appState = AppState()

    var body: some Scene {
        // Identified WindowGroup so the menu bar "Open" button can call
        // openWindow(id: "main") to recreate the main window after it has
        // been closed via the red traffic light (LSUIElement means close
        // button doesn't quit the app; without an id, SwiftUI's OpenWindow
        // action has no target to respawn).
        WindowGroup(id: "main") {
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
    @State private var showOnboarding = false

    var body: some View {
        MainWindowView()
            .sheet(isPresented: $showOnboarding) {
                OnboardingWindow {
                    showOnboarding = false
                }
                .environment(appState)
            }
            .onAppear {
                // Show onboarding on first launch (settings not yet loaded —
                // bootstrap() runs in .task; re-evaluate once it fires).
            }
            .onChange(of: appState.currentSettings.onboardingComplete) { _, newValue in
                if !newValue {
                    showOnboarding = true
                }
            }
            .task {
                // After bootstrap has loaded settings, decide whether to show onboarding.
                if !appState.currentSettings.onboardingComplete {
                    showOnboarding = true
                }
            }
    }
}
