// ModelPoolSection.swift
// macMLX
//
// Settings section exposing the v0.4 ModelPool "max resident memory"
// budget. Changing the stepper value both persists to
// `Settings.maxResidentMemoryGB` and pushes the new byte budget into
// `EngineCoordinator.setPoolBudget(bytes:)` so the live pool picks it
// up without a restart. Pin/unpin controls live on the Models tab's
// `LocalModelRow` instead — this section is only about the byte cap.

import SwiftUI
import MacMLXCore

struct ModelPoolSection: View {
    @Binding var maxResidentGB: Int

    var body: some View {
        Section("Model Pool") {
            HStack {
                Text("Max resident memory")
                Spacer()
                Stepper(
                    value: $maxResidentGB,
                    in: 2...256,
                    step: 1
                ) {
                    Text(String(maxResidentGB) + " GB")
                        .font(.system(.body, design: .monospaced))
                        .frame(minWidth: 80, alignment: .trailing)
                }
            }
            .help("When multiple loaded models exceed this, the least-recently-used non-pinned one is unloaded.")
        }
    }
}
