// DoneStep.swift
// macMLX — Onboarding Step 5

import SwiftUI

struct DoneStep: View {

    let onFinish: () async -> Void

    @State private var copied = false

    private let serverURL = "http://localhost:8000/v1"

    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .resizable()
                .scaledToFit()
                .frame(width: 64, height: 64)
                .foregroundStyle(.green)

            VStack(spacing: 8) {
                Text("You're all set!")
                    .font(.largeTitle.bold())

                Text("macMLX is running in your menu bar.\nThe inference server starts automatically when you load a model.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 8) {
                Text("Connect external tools:")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack {
                    Text(serverURL)
                        .font(.system(.body, design: .monospaced))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))

                    Button(copied ? "Copied!" : "Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(serverURL, forType: .string)
                        copied = true
                        Task {
                            try? await Task.sleep(for: .seconds(2))
                            copied = false
                        }
                    }
                    .buttonStyle(.bordered)
                }
            }

            Spacer()

            Button {
                Task { await onFinish() }
            } label: {
                Text("Start Using macMLX")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
    }
}

#Preview {
    DoneStep(onFinish: {})
        .frame(width: 480, height: 420)
}
