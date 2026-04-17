import ArgumentParser
import Foundation
import MacMLXCore

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run a model interactively (stdin REPL) or with a single prompt."
    )

    @Argument(help: "Model ID or display name to run.")
    var modelName: String

    @Argument(help: "Single prompt for non-interactive use.")
    var prompt: String?

    @Option(help: "Sampling temperature. Falls back to the persisted per-model value (GUI Parameters Inspector), then 0.7.")
    var temperature: Double?

    @Option(name: .customLong("max-tokens"), help: "Maximum tokens to generate. Falls back to the persisted per-model value, then 2048.")
    var maxTokens: Int?

    @Option(help: "System prompt. Falls back to the persisted per-model value.")
    var system: String?

    @Flag(name: .customLong("no-stream"), help: "Buffer the full response before printing.")
    var noStream = false

    @Flag(help: "Wrap output as JSON.")
    var json = false

    func run() async throws {
        let ctx = try await CLIContext.bootstrap()
        let models = try await ctx.library.scan(ctx.settings.modelDirectory)
        guard let local = models.first(where: {
            $0.id == modelName || $0.displayName == modelName
        }) else {
            throw ValidationError("Model not found: \(modelName). Run `macmlx list` to see available models.")
        }

        // Honour `settings.preferredEngine` — same engine choice the GUI
        // has. Deferred engines (SwiftLM / Python mlx-lm) throw CLIError.
        let engine = try ctx.makeEngine()
        print("Loading \(local.displayName)…")
        try await engine.load(local)
        print("Model ready.")

        // Layer CLI flags over persisted per-model overrides, so a user
        // who configured the model's temperature/top_p/system prompt in
        // the GUI sees those same defaults here unless they override
        // with explicit `--` flags.
        let (params, resolvedSystem) = await ctx.resolveParameters(
            for: local.id,
            explicitTemperature: temperature,
            explicitMaxTokens: maxTokens,
            explicitSystem: system,
            stream: !noStream
        )

        if let promptText = prompt {
            // Single-shot mode
            try await runSingle(
                prompt: promptText,
                engine: engine,
                model: local,
                params: params,
                system: resolvedSystem
            )
        } else if TTYDetect.isInteractive {
            // Interactive TUI/REPL mode — plain stdin REPL (see #18)
            try await ChatTUI.run(engine: engine, model: local, system: resolvedSystem)
        } else {
            // Non-interactive stdin mode: read lines until EOF
            try await runStdinLoop(
                engine: engine,
                model: local,
                params: params,
                system: resolvedSystem
            )
        }
    }

    // MARK: - Single-shot

    private func runSingle(
        prompt: String,
        engine: any InferenceEngine,
        model: LocalModel,
        params: GenerationParameters,
        system: String?
    ) async throws {
        let messages = [ChatMessage(role: .user, content: prompt)]
        let request = GenerateRequest(
            model: model.id,
            messages: messages,
            systemPrompt: system,
            parameters: params
        )

        var fullResponse = ""
        let stream = engine.generate(request)

        do {
            for try await chunk in stream {
                if !noStream && !json {
                    print(chunk.text, terminator: "")
                    fflush(stdout)
                }
                fullResponse += chunk.text
            }
        } catch {
            throw RuntimeError(error.localizedDescription)
        }

        if !noStream && !json {
            print()  // trailing newline
        }

        if json {
            let output: [String: String] = [
                "model": model.id,
                "prompt": prompt,
                "response": fullResponse,
            ]
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(output)
            FileHandle.standardOutput.write(data)
            print()
        } else if noStream {
            print(fullResponse)
        }
    }

    // MARK: - Stdin loop (piped / non-TTY)

    /// Read prompts from stdin line-by-line and respond to each.
    private func runStdinLoop(
        engine: any InferenceEngine,
        model: LocalModel,
        params: GenerationParameters,
        system: String?
    ) async throws {
        while let line = readLine(strippingNewline: true) {
            guard !line.isEmpty else { continue }

            let messages = [ChatMessage(role: .user, content: line)]
            let request = GenerateRequest(
                model: model.id,
                messages: messages,
                systemPrompt: system,
                parameters: params
            )

            var fullResponse = ""
            let stream = engine.generate(request)

            do {
                for try await chunk in stream {
                    if !json {
                        print(chunk.text, terminator: "")
                        fflush(stdout)
                    }
                    fullResponse += chunk.text
                }
            } catch {
                fputs("[Error: \(error.localizedDescription)]\n", stderr)
                continue
            }

            if json {
                let output: [String: String] = [
                    "model": model.id,
                    "prompt": line,
                    "response": fullResponse,
                ]
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(output)
                FileHandle.standardOutput.write(data)
                print()
            } else {
                print()  // trailing newline after streamed response
            }
        }
    }
}

// MARK: - Helpers

private struct RuntimeError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
