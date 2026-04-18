import SwiftUI

/// Collapsible disclosure for a `<think>…</think>` block. Collapsed by
/// default. While streaming, the label animates an ASCII braille
/// spinner; once the close tag arrives, it switches to a static
/// "thought" label.
struct ThinkBlockView: View {

    let content: String
    /// True only while the stream is still inside this think block. We
    /// animate the spinner only in that window; a completed thought is
    /// visually quieter.
    let isStreaming: Bool

    @State private var expanded: Bool = false
    @State private var animationFrame: Int = 0

    private static let spinnerFrames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    private static let frameInterval: Duration = .milliseconds(80)

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            disclosureHeader
            if expanded {
                Text(content)
                    .font(.caption)
                    .italic()
                    .foregroundStyle(.secondary)
                    .padding(.leading, 16)
                    .textSelection(.enabled)
            }
        }
        .task(id: isStreaming) {
            // Only the in-flight block needs the spinner animation.
            // .task(id:) cancels cleanly when isStreaming flips false.
            guard isStreaming else { return }
            while !Task.isCancelled {
                animationFrame = (animationFrame + 1) % Self.spinnerFrames.count
                try? await Task.sleep(for: Self.frameInterval)
            }
        }
    }

    private var disclosureHeader: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) { expanded.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.caption)
                Text(isStreaming
                     ? "\(Self.spinnerFrames[animationFrame]) thinking…"
                     : "thought")
                    .font(.caption)
                    .monospacedDigit()
            }
            .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    VStack(alignment: .leading, spacing: 12) {
        ThinkBlockView(
            content: "Let me reason about this step by step...",
            isStreaming: true
        )
        ThinkBlockView(
            content: "Let me reason about this step by step. The answer is 4.",
            isStreaming: false
        )
    }
    .padding()
    .frame(width: 420)
}
