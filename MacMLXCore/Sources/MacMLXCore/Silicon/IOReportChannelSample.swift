// Copyright © 2026 macMLX. English comments only.

/// One IOReport channel's delta over a sampling window, as a plain Swift value.
///
/// The IOReport calls hand back CoreFoundation dictionaries that are only valid
/// while the enclosing sample lives and can only be read on the sampling thread.
/// `IOReportReader` copies each channel into this `Sendable` snapshot so the pure
/// estimators (and W2's bottleneck classifier) can work with it off the sampling
/// path, in tests, and across actor boundaries.
public struct IOReportChannelSample: Sendable, Equatable {

    /// A single state's residency within a `.state`-format channel.
    public struct StateResidency: Sendable, Equatable {
        /// State label, e.g. `"OFF"`, `"P1"`, `"32GB/s"`.
        public let name: String
        /// Residency ticks accumulated in this state during the window.
        public let residency: UInt64

        public init(name: String, residency: UInt64) {
            self.name = name
            self.residency = residency
        }
    }

    /// IOReport group, e.g. `"Energy Model"`, `"GPU Stats"`, `"PMP0"`.
    public let group: String
    /// IOReport subgroup, e.g. `"GPU Performance States"`, `"DCS BW"`. Empty when the
    /// channel has none.
    public let subgroup: String
    /// Channel name, e.g. `"GPU"`, `"GPUPH"`, `"AMCC RD+WR"`.
    public let channel: String
    /// Value encoding of this channel.
    public let format: IOReportFormat
    /// Scalar delta for a `.simple` channel; `nil` for other formats.
    public let simpleValue: Int64?
    /// Ordered per-state residencies for a `.state` channel (index order preserved,
    /// which for bandwidth histograms means ascending bandwidth); `nil` otherwise.
    public let states: [StateResidency]?

    public init(
        group: String,
        subgroup: String,
        channel: String,
        format: IOReportFormat,
        simpleValue: Int64?,
        states: [StateResidency]?
    ) {
        self.group = group
        self.subgroup = subgroup
        self.channel = channel
        self.format = format
        self.simpleValue = simpleValue
        self.states = states
    }
}
