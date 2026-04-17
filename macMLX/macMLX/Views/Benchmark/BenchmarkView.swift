// BenchmarkView.swift
// macMLX
//
// Local benchmark tab — config form, last-result readout, history list.
// Issue #22.

import SwiftUI
import MacMLXCore

struct BenchmarkView: View {

    @Environment(AppState.self) private var appState

    // Read from AppState so an in-flight run survives tab switches
    // (same rationale as Chat — see AppState.benchmark docstring).
    var body: some View {
        BenchmarkContent(viewModel: appState.benchmark)
    }
}

// MARK: - Content

private struct BenchmarkContent: View {

    @Bindable var viewModel: BenchmarkViewModel
    @Environment(AppState.self) private var appState

    var body: some View {
        Form {
            configSection
            if let message = viewModel.errorMessage {
                Section {
                    Label(message, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .font(.caption)
                }
            }
            if let result = viewModel.lastResult {
                resultSection(result)
            }
            historySection
        }
        .formStyle(.grouped)
        .navigationTitle("Benchmark")
        .task {
            await viewModel.reload(modelDirectory: appState.currentSettings.modelDirectory)
        }
        .onChange(of: appState.currentSettings.modelDirectory) { _, newValue in
            Task { await viewModel.reload(modelDirectory: newValue) }
        }
    }

    // MARK: - Config

    private var configSection: some View {
        Section("Configuration") {
            Picker("Model", selection: $viewModel.selectedModel) {
                if viewModel.availableModels.isEmpty {
                    Text("No local models — download one in the Models tab")
                        .tag(LocalModel?.none)
                }
                ForEach(viewModel.availableModels) { model in
                    Text(model.id).tag(LocalModel?.some(model))
                }
            }
            .disabled(viewModel.isRunning)

            Stepper(value: $viewModel.promptTokens, in: 64...4096, step: 64) {
                LabeledContent("Prompt tokens", value: "\(viewModel.promptTokens)")
            }
            .disabled(viewModel.isRunning)

            Stepper(value: $viewModel.generationTokens, in: 32...2048, step: 32) {
                LabeledContent("Generation tokens", value: "\(viewModel.generationTokens)")
            }
            .disabled(viewModel.isRunning)

            Stepper(value: $viewModel.runs, in: 1...10) {
                LabeledContent("Runs", value: "\(viewModel.runs)")
            }
            .disabled(viewModel.isRunning)

            TextField("Notes (optional)", text: $viewModel.notes)
                .disabled(viewModel.isRunning)

            HStack {
                Spacer()
                if viewModel.isRunning {
                    ProgressView()
                        .controlSize(.small)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Cancel") {
                        viewModel.cancel()
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button {
                        viewModel.start()
                    } label: {
                        Label("Run Benchmark", systemImage: "stopwatch")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.selectedModel == nil)
                }
            }
        }
    }

    // MARK: - Last result

    private func resultSection(_ result: BenchmarkResult) -> some View {
        Section("Last result") {
            LabeledContent("Model", value: result.modelID)
            LabeledContent(
                "Hardware",
                value: "\(result.system.chip.replacingOccurrences(of: "Apple ", with: ""))  \u{00B7}  \(result.system.ramGB) GiB"
            )
            metricRow("Prefill", value: result.promptTPS, unit: "tok/s", digits: 0)
            metricRow("Generation", value: result.generationTPS, unit: "tok/s", digits: 1)
            metricRow("Time to first token", value: result.ttftMs, unit: "ms", digits: 0)
            if result.memoryUsedGB > 0 {
                metricRow("Peak memory", value: result.memoryUsedGB, unit: "GB", digits: 2)
            }
            if result.modelLoadTimeS > 0 {
                metricRow("Load time", value: result.modelLoadTimeS, unit: "s", digits: 1)
            }

            HStack {
                Button {
                    if let url = ShareURLBuilder.communityShareURL(for: result) {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    Label("Share to Community", systemImage: "arrowshape.turn.up.right")
                }
                .buttonStyle(.bordered)

                Button {
                    copyJSON(result)
                } label: {
                    Label("Copy as JSON", systemImage: "doc.on.clipboard")
                }
                .buttonStyle(.bordered)

                Spacer()
            }
        }
    }

    private func metricRow(
        _ label: String,
        value: Double,
        unit: String,
        digits: Int
    ) -> some View {
        LabeledContent(label) {
            Text(formatted(value, digits: digits))
                .font(.system(.body, design: .monospaced))
            + Text(" \(unit)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func formatted(_ value: Double, digits: Int) -> String {
        String(format: "%.\(digits)f", value)
    }

    private func copyJSON(_ result: BenchmarkResult) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(result),
              let json = String(data: data, encoding: .utf8) else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(json, forType: .string)
    }

    // MARK: - History

    private var historySection: some View {
        Section {
            if viewModel.history.isEmpty {
                Text("No prior runs.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.history) { result in
                    historyRow(result)
                }
            }
        } header: {
            HStack {
                Text("History")
                Spacer()
                if !viewModel.history.isEmpty {
                    Button("Clear") {
                        Task { await viewModel.clearHistory() }
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                }
            }
        }
    }

    private func historyRow(_ result: BenchmarkResult) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(result.modelID)
                    .font(.body)
                Text("\(result.system.chip.replacingOccurrences(of: "Apple ", with: ""))  \u{00B7}  \(result.system.ramGB) GiB  \u{00B7}  \(result.timestamp.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(formatted(result.generationTPS, digits: 1)) tok/s")
                .font(.system(.body, design: .monospaced))
            Button {
                Task { await viewModel.delete(id: result.id) }
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Share URL builder

/// Builds a pre-filled GitHub-issue URL for submitting a benchmark to the
/// community leaderboard pipeline (issue #22). The target repo provides an
/// issue template named `benchmark_submission.yml` with a single textarea
/// `data` that accepts the result JSON.
///
/// Keeping this as a namespace (not a method on the view model) so it's
/// easy to unit-test from MacMLXCoreTests in a follow-up commit.
enum ShareURLBuilder {

    static let repoPath = "magicnight/Mac-MLX"

    static func communityShareURL(for result: BenchmarkResult) -> URL? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(result),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }

        // GitHub's new-issue URL takes `template=<file>` + one key per
        // template input. Our template exposes a `data` textarea for the
        // full JSON payload. We URL-encode once; GitHub decodes it on
        // the issue form.
        guard var components = URLComponents(string: "https://github.com/\(repoPath)/issues/new") else {
            return nil
        }
        components.queryItems = [
            URLQueryItem(name: "template", value: "benchmark_submission.yml"),
            URLQueryItem(name: "labels", value: "benchmark"),
            URLQueryItem(name: "title", value: "[benchmark] \(result.modelID) on \(result.system.chip)"),
            URLQueryItem(name: "data", value: "```json\n\(json)\n```"),
        ]
        return components.url
    }
}

#Preview {
    BenchmarkView()
        .environment(AppState())
        .frame(width: 700, height: 500)
}
