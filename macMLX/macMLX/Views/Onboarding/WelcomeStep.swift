// WelcomeStep.swift
// macMLX — Onboarding Step 1

import SwiftUI

struct WelcomeStep: View {

    let onGetStarted: () -> Void
    let onSkip: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            // App icon / logo area
            Image(systemName: "cpu.fill")
                .resizable()
                .scaledToFit()
                .frame(width: 80, height: 80)
                .foregroundStyle(.tint)

            VStack(spacing: 8) {
                Text("Welcome to macMLX")
                    .font(.largeTitle.bold())

                Text("Local LLM inference, native on your Mac.\nPowered by Apple MLX · No cloud required.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .font(.body)
            }

            Spacer()

            VStack(spacing: 12) {
                Button(action: onGetStarted) {
                    Label("Get Started", systemImage: "arrow.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                Button("Skip setup and open app", action: onSkip)
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            }
        }
        .padding(40)
    }
}

#Preview {
    WelcomeStep(onGetStarted: {}, onSkip: {})
        .frame(width: 480, height: 400)
}
