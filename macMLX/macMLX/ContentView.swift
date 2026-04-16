import SwiftUI
import MacMLXCore

struct ContentView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "cpu")
                .font(.system(size: 48))
                .foregroundStyle(.tint)
            Text("macMLX")
                .font(.largeTitle)
            Text("Core \(MacMLXCore.version)")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 320, minHeight: 200)
        .padding()
    }
}

#Preview {
    ContentView()
}
