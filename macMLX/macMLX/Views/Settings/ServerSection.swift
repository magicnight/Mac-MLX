// ServerSection.swift
// macMLX
//
// Form section for HTTP server configuration.

import SwiftUI
import MacMLXCore

struct ServerSection: View {

    @Binding var serverPort: Int
    @Binding var autoStartServer: Bool

    var body: some View {
        Section("HTTP Server") {
            HStack {
                Text("Port")
                Spacer()
                Stepper(
                    value: $serverPort,
                    in: 1024...65535,
                    step: 1
                ) {
                    // Use `String(...)` not `"\(serverPort)"` — SwiftUI's
                    // Text localises Int interpolation and inserts a
                    // thousand separator (8,000 instead of 8000).
                    Text(String(serverPort))
                        .font(.system(.body, design: .monospaced))
                        .frame(minWidth: 50, alignment: .trailing)
                }
            }

            Toggle("Auto-start server on launch", isOn: $autoStartServer)

            HStack {
                Text("Base URL")
                    .foregroundStyle(.secondary)
                Spacer()
                Text("http://localhost:" + String(serverPort) + "/v1")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }
}

#Preview {
    Form {
        ServerSection(
            serverPort: .constant(8000),
            autoStartServer: .constant(false)
        )
    }
    .formStyle(.grouped)
    .frame(width: 500)
}
