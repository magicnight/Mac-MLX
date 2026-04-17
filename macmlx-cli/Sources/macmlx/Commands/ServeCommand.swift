import ArgumentParser
import Darwin
import Foundation
import MacMLXCore

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the OpenAI-compatible inference server."
    )

    @Option(help: "Model ID or display name to load at startup.")
    var model: String?

    @Option(help: "Port to listen on (default: 8000).")
    var port: Int = 8000

    @Flag(name: .long, help: "Disable TUI dashboard and use plain stdout.")
    var noTui = false

    func run() async throws {
        let ctx = try await CLIContext.bootstrap()
        // Honour `settings.preferredEngine` — keeps CLI + GUI consistent.
        let engine = try ctx.makeEngine()

        if let modelName = model {
            let models = try await ctx.library.scan(ctx.settings.modelDirectory)
            guard let local = models.first(where: {
                $0.id == modelName || $0.displayName == modelName
            }) else {
                throw ValidationError("Model not found: \(modelName). Run `macmlx list` to see available models.")
            }
            print("Loading model \(local.displayName)…")
            try await engine.load(local)
            print("Model loaded.")
        }

        // Cold-swap (v0.3.3): resolver lets the server pull any
        // locally-downloaded model into memory on demand when an
        // inbound `/v1/chat/completions` names an unloaded model.
        let library = ctx.library
        let modelDirectory = ctx.settings.modelDirectory
        let resolver: HummingbirdServer.ModelResolver = { modelID in
            let models = (try? await library.scan(modelDirectory)) ?? []
            return models.first { $0.id == modelID || $0.displayName == modelID }
        }
        let server = HummingbirdServer(engine: engine, modelResolver: resolver)
        let actualPort = try await server.start(preferredPort: port)

        let startedAt = Date()
        let record = PIDFile.Record(
            pid: getpid(),
            port: actualPort,
            modelID: model,
            startedAt: startedAt
        )
        try PIDFile.write(record)

        // Register SIGINT / SIGTERM handlers for graceful shutdown.
        // SwiftTUI intercepts SIGINT internally, but in plain mode we need
        // to handle it ourselves.
        let stopFlag = UnsafeSendable(DispatchSemaphore(value: 0))
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)

        let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        sigintSource.setEventHandler {
            stopFlag.value.signal()
        }
        sigintSource.resume()

        let sigtermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        sigtermSource.setEventHandler {
            stopFlag.value.signal()
        }
        sigtermSource.resume()

        // Print status regardless of TUI mode (TUI is plain stdout in v0.1)
        ServeDashboard.printStatus(port: actualPort, modelID: model ?? "(none)")

        // Wait for signal
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                stopFlag.value.wait()
                continuation.resume()
            }
        }

        print("\nStopping server…")
        try? PIDFile.clear()
        await server.stop()
        print("Server stopped.")
    }
}

// MARK: - Helpers

/// Wrapper to pass a non-Sendable value across async boundaries under Swift 6.
private struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}
