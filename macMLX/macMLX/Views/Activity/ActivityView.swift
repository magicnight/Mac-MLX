// ActivityView.swift
// macMLX
//
// The v0.7 W3 silicon-metrics observation panel: a main-window detail tab that
// shows, live, what the Apple Silicon is doing during inference and what — if
// anything — is limiting throughput.
//
// Design follows .claude/ui-guidelines.md to the letter: native macOS containers
// (GroupBox), system fonts, semantic colors used sparingly as status accents,
// automatic dark mode, monospaced digits so numbers don't jitter as they update.
//
// HONESTY IS A FIRST-CLASS UI CONCERN HERE (matching the W1/W2 data model):
//   * memory-bandwidth readings are labelled "estimated", and "saturated" when the
//     top band filled (the real value may be higher);
//   * the ANE shows a POWER PROXY only, explicitly labelled — never a utilization %,
//     because no ANE occupancy counter exists;
//   * the Media Engine is not shown at all (IOReport gives it no duty cycle);
//   * when IOReport is unavailable the IOReport-derived readings show "unavailable"
//     with the reason, rather than fabricated zeros — thermal and memory pressure,
//     which come from public APIs, still display;
//   * a bottleneck verdict that rests on estimated bandwidth is flagged as such, so
//     it never reads as a measured certainty;
//   * with no generation running, the bottleneck area says so instead of showing a
//     stale attribution.

import SwiftUI
import MacMLXCore

struct ActivityView: View {

    @Environment(AppState.self) private var appState

    var body: some View {
        ActivityContent(monitor: appState.siliconMonitor)
    }
}

// MARK: - Content

private struct ActivityContent: View {

    /// Read-only; the panel binds the monitor's @Observable state and never mutates
    /// it beyond the sampling lifecycle calls below.
    let monitor: SiliconMonitor

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                statusHeader
                bottleneckCard
                throughputCard
                hardwareCard
                powerCard
                if monitor.ioReportAvailable == false {
                    ioReportUnavailableNote
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .navigationTitle("Activity")
        // Sampling is gated on visibility: poll IOReport only while the user is
        // watching this tab (the phase-event stream keeps running elsewhere).
        .onAppear { monitor.startSampling() }
        .onDisappear { monitor.stopSampling() }
    }

    // MARK: - Status header ("Active Now")

    private var statusHeader: some View {
        HStack(spacing: 10) {
            StatusDot(active: monitor.isGenerating, pulses: !reduceMotion)
            VStack(alignment: .leading, spacing: 1) {
                Text(monitor.isGenerating ? "Generating" : "Idle")
                    .font(.headline)
                Text(activitySubtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if let phase = monitor.phase {
                PhaseBadge(phase: phase)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: monitor.isGenerating)
    }

    private var activitySubtitle: String {
        guard monitor.isGenerating else { return "No active generation" }
        switch monitor.phase {
        case .prefill: return "Processing the prompt"
        case .decode: return "Producing tokens"
        case nil: return "In flight"
        }
    }

    // MARK: - Bottleneck

    private var bottleneckCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                if let verdict = monitor.verdict {
                    HStack(spacing: 8) {
                        BottleneckBadge(category: verdict.category)
                        PhaseBadge(phase: verdict.phase)
                        Spacer()
                    }
                    Text(verdict.advice)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if verdict.restsOnEstimatedBandwidth {
                        Label(
                            "Attribution rests on estimated bandwidth, not a measured value.",
                            systemImage: "info.circle"
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                } else if monitor.isGenerating {
                    // Generating, but the verdict is deliberately withheld until the
                    // classifier's rolling window sheds the previous run's tail (so a
                    // stale, possibly alarming verdict never flashes at the start of a
                    // run). Show that it is measuring — NOT "no active generation",
                    // which would contradict the "Generating" status header.
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Measuring — establishing a baseline for this generation…")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    // Idle: honest placeholder rather than a stale verdict.
                    HStack(spacing: 8) {
                        Image(systemName: "moon.zzz")
                            .foregroundStyle(.secondary)
                        Text("No active generation — start a chat or benchmark to see the live bottleneck.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        } label: {
            cardLabel("Bottleneck", systemImage: "bolt.trianglebadge.exclamationmark")
        }
    }

    // MARK: - Throughput (PP / TG)

    private var throughputCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 24) {
                    ThroughputReadout(
                        title: "Prefill",
                        caption: "PP",
                        tokensPerSecond: monitor.prefillTokensPerSecond
                    )
                    Divider().frame(height: 44)
                    ThroughputReadout(
                        title: "Decode",
                        caption: "TG",
                        tokensPerSecond: monitor.decodeTokensPerSecond
                    )
                    Spacer()
                }
                // Always-visible honesty label: these are real counts ÷ real time
                // from the last COMPLETED run, not a live per-token rate (which the
                // engine deliberately does not sample on the decode hot path).
                Text("Last completed generation — real counts over elapsed time, not a live rate.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(4)
        } label: {
            cardLabel("Throughput", systemImage: "speedometer")
        }
    }

    // MARK: - Hardware

    private var hardwareCard: some View {
        GroupBox {
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 240), spacing: 16)],
                alignment: .leading,
                spacing: 16
            ) {
                gpuTile
                bandwidthTile
                thermalTile
                memoryTile
            }
            .padding(4)
        } label: {
            cardLabel("Hardware", systemImage: "cpu")
        }
    }

    @ViewBuilder
    private var gpuTile: some View {
        if let gpu = monitor.latestSample?.gpuUtilization {
            MetricGauge(
                title: "GPU utilization",
                value: gpu.busyFraction,
                display: percent(gpu.busyFraction),
                tint: .accentColor,
                caption: "occupancy — real performance-state residency"
            )
        } else {
            UnavailableTile(title: "GPU utilization", reason: ioReportGate)
        }
    }

    @ViewBuilder
    private var bandwidthTile: some View {
        // Prefer the GPU (AGX) requestor — the one that matters for MLX inference;
        // fall back to the coarse DRAM aggregate only when AGX is absent.
        if let bw = monitor.latestSample?.gpuBandwidth {
            BandwidthTile(title: "GPU memory bandwidth", sample: bw, coarse: false)
        } else if let bw = monitor.latestSample?.dramBandwidth {
            BandwidthTile(title: "DRAM bandwidth (aggregate)", sample: bw, coarse: true)
        } else {
            UnavailableTile(title: "Memory bandwidth", reason: ioReportGate)
        }
    }

    private var thermalTile: some View {
        // Thermal pressure is a public API — always present, even without IOReport.
        let thermal = monitor.latestSample?.thermalPressure
        return StatusTile(
            title: "Thermal pressure",
            valueText: thermal.map(thermalLabel) ?? "—",
            tint: thermal.map(thermalTint) ?? .secondary,
            caption: "ProcessInfo.thermalState"
        )
    }

    @ViewBuilder
    private var memoryTile: some View {
        // Memory pressure is also public — present without IOReport.
        if let mem = monitor.latestSample?.memoryPressure {
            MetricGauge(
                title: "Memory pressure",
                value: mem.usedFraction,
                display: "\(percent(mem.usedFraction)) · \(memoryLevelLabel(mem.level))",
                tint: memoryTint(mem.level),
                caption: "non-reclaimable working set of \(bytesGiB(mem.totalBytes)) GiB"
            )
        } else {
            UnavailableTile(title: "Memory pressure", reason: "kernel read failed")
        }
    }

    // MARK: - Power

    private var powerCard: some View {
        GroupBox {
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 150), spacing: 16)],
                alignment: .leading,
                spacing: 12
            ) {
                PowerReadout(title: "CPU", watts: monitor.latestSample?.cpuPower?.watts)
                PowerReadout(title: "GPU", watts: monitor.latestSample?.gpuPower?.watts)
                PowerReadout(title: "DRAM", watts: monitor.latestSample?.dramPower?.watts)
                PowerReadout(
                    title: "ANE",
                    watts: monitor.latestSample?.anePower?.watts,
                    footnote: "power proxy — no utilization signal"
                )
            }
            .padding(4)
        } label: {
            cardLabel("Power", systemImage: "bolt")
                .help("Per-rail average power over the sampling window. The ANE exposes only power, never an occupancy percentage.")
        }
    }

    // MARK: - IOReport-unavailable note

    private var ioReportUnavailableNote: some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text("IOReport unavailable")
                    .font(.callout.weight(.medium))
                Text(monitor.ioReportUnavailableReason
                    ?? "GPU utilization, memory bandwidth and per-rail power need IOReport.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } icon: {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Shared bits

    /// The reason string shown on IOReport-derived tiles: distinguishes "sensors
    /// still starting" (no sample yet) from a genuine IOReport-unavailable state.
    private var ioReportGate: String {
        switch monitor.ioReportAvailable {
        case .none: return "starting sensors…"
        case .some(false): return "IOReport unavailable"
        case .some(true): return "no reading this window"
        }
    }

    private func cardLabel(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.headline)
    }

    // MARK: - Formatting helpers

    private func percent(_ fraction: Double) -> String {
        "\(Int((fraction * 100).rounded()))%"
    }

    private func bytesGiB(_ bytes: UInt64) -> String {
        String(format: "%.0f", Double(bytes) / 1_073_741_824)
    }

    private func thermalLabel(_ t: ThermalPressure) -> String {
        switch t {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        }
    }

    private func thermalTint(_ t: ThermalPressure) -> Color {
        switch t {
        case .nominal, .fair: return .secondary
        case .serious: return .orange
        case .critical: return .red
        }
    }

    private func memoryLevelLabel(_ level: MemoryPressureSample.Level) -> String {
        switch level {
        case .unknown: return "Unknown"
        case .normal: return "Normal"
        case .warning: return "Warning"
        case .critical: return "Critical"
        }
    }

    private func memoryTint(_ level: MemoryPressureSample.Level) -> Color {
        switch level {
        case .warning: return .orange
        case .critical: return .red
        case .normal, .unknown: return .accentColor
        }
    }
}

// MARK: - Status dot

private struct StatusDot: View {
    let active: Bool
    let pulses: Bool
    @State private var pulse = false

    var body: some View {
        Circle()
            .fill(active ? Color.green : Color.gray)
            .frame(width: 10, height: 10)
            .opacity(active && pulses && pulse ? 0.35 : 1)
            .animation(
                active && pulses
                    ? .easeInOut(duration: 1).repeatForever(autoreverses: true)
                    : .default,
                value: pulse
            )
            .onAppear { pulse = true }
            .accessibilityLabel(active ? "Generating" : "Idle")
    }
}

// MARK: - Phase badge

private struct PhaseBadge: View {
    let phase: InferencePhase

    var body: some View {
        Text(phase == .prefill ? "Prefill" : "Decode")
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.accentColor.opacity(0.15), in: Capsule())
            .foregroundStyle(Color.accentColor)
    }
}

// MARK: - Bottleneck badge

private struct BottleneckBadge: View {
    let category: BottleneckVerdict.Category

    var body: some View {
        Label(title, systemImage: symbol)
            .font(.callout.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(tint.opacity(0.15), in: Capsule())
            .foregroundStyle(tint)
    }

    private var title: String {
        switch category {
        case .normal: return "Balanced"
        case .memoryBound: return "Memory-bound"
        case .thermalThrottled: return "Thermal-throttled"
        case .computeBound: return "Compute-bound"
        case .bandwidthBound: return "Bandwidth-bound"
        }
    }

    private var symbol: String {
        switch category {
        case .normal: return "checkmark.seal"
        case .memoryBound: return "memorychip"
        case .thermalThrottled: return "thermometer.high"
        case .computeBound: return "cpu"
        case .bandwidthBound: return "arrow.left.arrow.right"
        }
    }

    /// Memory/thermal are the actionable-trouble states → warm accent. Compute /
    /// bandwidth are the expected healthy regimes → neutral accent. Balanced → green.
    private var tint: Color {
        switch category {
        case .normal: return .green
        case .memoryBound: return .red
        case .thermalThrottled: return .orange
        case .computeBound, .bandwidthBound: return .accentColor
        }
    }
}

// MARK: - Throughput readout

private struct ThroughputReadout: View {
    let title: String
    let caption: String
    let tokensPerSecond: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(valueText)
                    .font(.system(.title2, design: .rounded))
                    .fontWeight(.semibold)
                    .monospacedDigit()
                Text("tok/s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .accessibilityElement(children: .combine)
    }

    private var valueText: String {
        guard let tps = tokensPerSecond else { return "—" }
        return String(format: "%.1f", tps)
    }
}

// MARK: - Metric gauge tile

private struct MetricGauge: View {
    let title: String
    let value: Double
    let display: String
    let tint: Color
    let caption: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.subheadline)
                Spacer()
                Text(display)
                    .font(.system(.subheadline, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Gauge(value: value.clamped01) { EmptyView() }
                .gaugeStyle(.accessoryLinearCapacity)
                .tint(tint)
                .animation(.easeInOut(duration: 0.25), value: value)
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title): \(display)")
    }
}

// MARK: - Bandwidth tile (estimated, honesty-annotated)

private struct BandwidthTile: View {
    let title: String
    let sample: MemoryBandwidthSample
    /// The coarse DRAM aggregate (±16 GB/s quantum) vs. the fine 1-GB/s-band GPU
    /// channel — drives the resolution wording.
    let coarse: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.subheadline)
                Spacer()
                HStack(alignment: .firstTextBaseline, spacing: 3) {
                    Text(String(format: "%.0f", sample.estimatedGBPerSecond))
                        .font(.system(.subheadline, design: .rounded))
                        .monospacedDigit()
                    Text("GB/s")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 6) {
                // Every reading is a residency-weighted estimate — always labelled.
                Tag(text: "estimated", tint: .secondary)
                if sample.isSaturated {
                    Tag(text: "saturated · may be higher", tint: .orange)
                }
            }
            Text(coarse
                ? "±16 GB/s quantum — cannot resolve traffic below ~32 GB/s"
                : "1 GB/s-band resolution (AGX requestor)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Status tile (thermal)

private struct StatusTile: View {
    let title: String
    let valueText: String
    let tint: Color
    let caption: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.subheadline)
                Spacer()
                Text(valueText)
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(tint)
            }
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title): \(valueText)")
    }
}

// MARK: - Power readout

private struct PowerReadout: View {
    let title: String
    let watts: Double?
    var footnote: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(valueText)
                    .font(.system(.title3, design: .rounded))
                    .monospacedDigit()
                Text("W")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let footnote {
                Text(footnote)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(footnote == nil
            ? "\(title) power: \(valueText) watts"
            : "\(title) power: \(valueText) watts, \(footnote ?? "")")
    }

    private var valueText: String {
        guard let watts else { return "—" }
        return String(format: "%.1f", watts)
    }
}

// MARK: - Unavailable tile

private struct UnavailableTile: View {
    let title: String
    let reason: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.subheadline)
                Spacer()
                Text("—")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title): unavailable, \(reason)")
    }
}

// MARK: - Small tag

private struct Tag: View {
    let text: String
    let tint: Color

    var body: some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tint.opacity(0.12), in: Capsule())
            .foregroundStyle(tint)
    }
}

// MARK: - Utilities

private extension Double {
    /// Clamp to the gauge's valid 0…1 range (residency-weighted fractions are
    /// already bounded, but a defensive clamp keeps Gauge from asserting).
    var clamped01: Double { Swift.max(0, Swift.min(1, self)) }
}

#Preview {
    ActivityView()
        .environment(AppState())
        .frame(width: 720, height: 640)
}
