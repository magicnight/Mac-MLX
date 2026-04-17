// ChatTUI.swift
// macmlx
//
// Plain stdin/stdout REPL for `macmlx run` in interactive mode. No
// full-screen TUI — a readline-style loop is the right level of
// complexity for CLI chat. The SwiftTUI experiment was removed in
// v0.3.5 (upstream unmaintained + Swift 6 incompatible).

import Foundation
import MacMLXCore

enum ChatTUI {
    /// Run an interactive stdin chat loop with `engine` and `model`.
    ///
    /// Reads lines from stdin (via `readLine()`), generates responses, and
    /// streams output to stdout. Exits on EOF or empty input.
    static func run(
        engine: any InferenceEngine,
        model: LocalModel,
        system: String?
    ) async throws {
        let width = 54
        print(CLITerm.boxHeader("macmlx run — \(model.displayName)", width: width))
        print(CLITerm.colourise("Type a message and press Enter. Empty line or Ctrl+D to quit.", CLITerm.dim))
        print(CLITerm.boxFooter(width: width))

        var conversationMessages: [ChatMessage] = []

        while true {
            print(CLITerm.colourise("> ", CLITerm.cyan), terminator: "")
            fflush(stdout)
            guard let line = readLine(strippingNewline: true), !line.isEmpty else {
                print(CLITerm.colourise("Goodbye.", CLITerm.dim))
                break
            }

            conversationMessages.append(ChatMessage(role: .user, content: line))

            let request = GenerateRequest(
                model: model.id,
                messages: conversationMessages,
                systemPrompt: system,
                parameters: GenerationParameters()
            )

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
                print(CLITerm.colourise("\n[Error: \(error.localizedDescription)]", CLITerm.red))
            }

            conversationMessages.append(ChatMessage(role: .assistant, content: responseText))
        }
    }
}
