// SettingsView.swift
// macMLX
//
// Settings form with General, Engine, and Server sections.
// Replaces the Stage 4 Task 7 stub.

import SwiftUI
import MacMLXCore

struct SettingsView: View {

    @Environment(AppState.self) private var appState
    @State private var showOnboarding = false

    // Local copies of settings for binding — flushed on each change
    @State private var modelDirectory: URL = .init(filePath: "")
    @State private var selectedEngine: EngineID = .mlxSwift
    @State private var serverPort: Int = 8000
    @State private var autoStartServer: Bool = false

    var body: some View {
        Form {
            generalSection
            EnginePickerSection(
                selectedEngine: $selectedEngine,
                onEngineChange: { engine in
                    Task {
                        await appState.updateSettings { $0.preferredEngine = engine }
                        appState.coordinator.switchTo(engine)
                    }
                }
            )
            ServerSection(
                serverPort: $serverPort,
                autoStartServer: $autoStartServer
            )
            .onChange(of: serverPort) { _, newValue in
                Task { await appState.updateSettings { $0.serverPort = newValue } }
            }
            .onChange(of: autoStartServer) { _, newValue in
                Task { await appState.updateSettings { $0.autoStartServer = newValue } }
            }

            rerunSetupSection
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .onAppear { syncFromSettings() }
        .onChange(of: appState.currentSettings) { _, _ in syncFromSettings() }
        .sheet(isPresented: $showOnboarding) {
            OnboardingWindow {
                showOnboarding = false
            }
            .environment(appState)
        }
    }

    // MARK: - General section

    private var generalSection: some View {
        Section("General") {
            HStack {
                Text("Model Directory")
                Spacer()
                Text(modelDirectory.path(percentEncoded: false)
                    .replacingOccurrences(
                        of: FileManager.default.homeDirectoryForCurrentUser.path,
                        with: "~"
                    ))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)

                Button("Browse…") { showModelDirectoryPicker() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    // MARK: - Re-run setup section

    private var rerunSetupSection: some View {
        Section {
            Button("Re-run Setup Wizard…") {
                // Reset onboarding flag so the wizard shows fresh
                Task {
                    await appState.updateSettings { $0.onboardingComplete = false }
                    showOnboarding = true
                }
            }
            .foregroundStyle(Color.accentColor)
        } footer: {
            Text("Opens the first-launch setup wizard. All current settings are preserved.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Helpers

    private func syncFromSettings() {
        let s = appState.currentSettings
        modelDirectory = s.modelDirectory
        selectedEngine = s.preferredEngine
        serverPort = s.serverPort
        autoStartServer = s.autoStartServer
    }

    private func showModelDirectoryPicker() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Choose Model Directory"
        if panel.runModal() == .OK, let url = panel.url {
            modelDirectory = url
            Task { await appState.updateSettings { $0.modelDirectory = url } }
        }
    }
}

#Preview {
    SettingsView()
        .environment(AppState())
        .frame(width: 600, height: 500)
}
