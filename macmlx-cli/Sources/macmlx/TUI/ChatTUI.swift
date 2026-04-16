import Foundation
import MacMLXCore
import SwiftTUI

// TODO: v0.2 — SwiftTUI's View protocol is nonisolated, conflicting with
// @MainActor state classes under Swift 6 strict concurrency. Full TUI deferred.
// For v0.1 the interactive chat uses a plain readline-style stdin loop.

/// Entry point for the interactive chat TUI.
///
/// v0.1: Uses a plain stdin/stdout REPL. SwiftTUI rendering deferred to v0.2.
enum ChatTUI {
    /// Run an interactive stdin chat loop with `engine` and `model`.
    ///
    /// Reads lines from stdin (via `readLine()`), generates responses, and
    /// streams output to stdout. Exits on EOF or empty input.
    static func run(
        engine: MLXSwiftEngine,
        model: LocalModel,
        system: String?
    ) async throws {
        print("macmlx run — \(model.displayName)")
        print("Type your message and press Enter. Empty line or Ctrl+D to quit.")
        print(String(repeating: "─", count: 50))

        var conversationMessages: [ChatMessage] = []

        while true {
            print("> ", terminator: "")
            fflush(stdout)
            guard let line = readLine(strippingNewline: true), !line.isEmpty else {
                print("\nGoodbye.")
                break
            }

            conversationMessages.append(ChatMessage(role: .user, content: line))

            let request = GenerateRequest(
                model: model.id,
                messages: conversationMessages,
                systemPrompt: system,
                parameters: GenerationParameters()
            )

            print("", terminator: "")  // assistant response on new line
            var responseText = ""

            do {
                let stream = engine.generate(request)
                for try await chunk in stream {
                    print(chunk.text, terminator: "")
                    fflush(stdout)
                    responseText += chunk.text
                }
                print()  // newline after response
            } catch {
                print("\n[Error: \(error.localizedDescription)]")
            }

            conversationMessages.append(ChatMessage(role: .assistant, content: responseText))
        }
    }
}

// Minimal SwiftTUI stub — keeps the product linked
// TODO: v0.2 — replace with a real Application(rootView:).start() chat dashboard
private struct _ChatTUIView: View {
    var body: some View {
        Text("macmlx run — TUI v0.2")
    }
}
