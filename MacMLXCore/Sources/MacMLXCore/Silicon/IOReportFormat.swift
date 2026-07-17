// Copyright © 2026 macMLX. English comments only.

/// Value encoding of an IOReport channel, mirroring the C `MacMLXIOReportFormat`.
///
/// Kept as a first-class Swift enum (rather than leaning on the imported C enum) so
/// the sampling code reads cleanly and does not depend on Clang's prefix-stripping.
/// Only `.simple` and `.state` are consumed today; the rest are modelled so an
/// unexpected format is surfaced as `.other` instead of being misread as a counter.
public enum IOReportFormat: Int32, Sendable, Equatable {
    /// Not a readable channel (or IOReport unavailable).
    case invalid = 0
    /// A single monotonic counter — in a delta it is the increase over the window
    /// (millijoules for the Energy Model group, bytes for AMC counters).
    case simple = 1
    /// Per-state residency ticks — a delta gives ticks spent in each state during
    /// the window, so a state's fraction of the row total is its fraction of time.
    case state = 2
    /// A distribution the samplers here do not consume.
    case histogram = 3
    /// An array of simple counters the samplers here do not consume.
    case simpleArray = 4
    /// Any format not enumerated above.
    case other = -1

    /// Map a raw format code from `MacMLXIOReportChannelGetFormat`.
    public init(rawFormat: Int32) {
        self = IOReportFormat(rawValue: rawFormat) ?? .other
    }
}
