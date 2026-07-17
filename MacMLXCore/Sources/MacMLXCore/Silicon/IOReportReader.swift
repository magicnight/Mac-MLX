// Copyright © 2026 macMLX. English comments only.

import CIOReport
import CoreFoundation
import Foundation

/// The one impure boundary of the silicon layer: owns a live IOReport session and
/// turns each poll into `Sendable` `IOReportChannelSample` value snapshots.
///
/// Deliberately a non-`Sendable` `final class`. It holds an opaque session pointer and
/// must be serialised; `IOReportSiliconSampler` (an actor) owns exactly one and never
/// shares it, which provides that serialisation. The session pointer, the rolling
/// previous sample, and the interval clock all live in C (see the CIOReport session
/// API), so this type only marshals CoreFoundation reads into Swift values.
///
/// If IOReport is unavailable (older/newer OS, symbol renamed, sandbox) the reader
/// constructs successfully with `isAvailable == false` and `poll()` returns `nil`, so
/// callers degrade to "metrics unavailable" instead of failing.
final class IOReportReader {

    /// The (group, subgroup) channel sets the samplers need. Narrowing the
    /// subscription to just these keeps each poll cheap versus the ~10k channels the
    /// machine advertises.
    ///   * Energy Model — CPU/GPU/ANE/DRAM power (Simple, millijoules).
    ///   * GPU Stats / GPU Performance States — real GPU occupancy (State residency).
    ///   * PMP0 / DCS BW — per-requestor DRAM bandwidth histograms (State residency).
    private static let wantedGroups: [(group: String, subgroup: String?)] = [
        ("Energy Model", nil),
        ("GPU Stats", "GPU Performance States"),
        ("PMP0", "DCS BW"),
    ]

    let isAvailable: Bool
    let unavailableReason: String?

    private let session: OpaquePointer?

    init() {
        guard MacMLXIOReportIsAvailable() else {
            self.isAvailable = false
            self.unavailableReason =
                MacMLXIOReportUnavailableReason().map { String(cString: $0) }
                ?? "IOReport unavailable"
            self.session = nil
            return
        }

        // Build the desired-channel dictionary by copying each wanted group and
        // merging them into the first one.
        var desired: CFMutableDictionary?
        for spec in Self.wantedGroups {
            let subgroup = spec.subgroup.map { $0 as CFString }
            guard let group = MacMLXIOReportCopyChannelsInGroup(
                spec.group as CFString, subgroup
            ) else {
                continue  // group absent on this chip; skip it
            }
            if let accumulator = desired {
                MacMLXIOReportMergeChannels(accumulator, group)
            } else {
                desired = group
            }
        }

        guard let desired else {
            self.isAvailable = false
            self.unavailableReason = "no requested IOReport channels present"
            self.session = nil
            return
        }

        guard let opened = MacMLXIOReportOpen(desired) else {
            self.isAvailable = false
            self.unavailableReason = "IOReport subscription was rejected"
            self.session = nil
            return
        }

        self.session = opened
        self.isAvailable = true
        self.unavailableReason = nil
    }

    deinit {
        if let session {
            MacMLXIOReportClose(session)
        }
    }

    /// Sample now and return the per-channel deltas since the previous poll, plus the
    /// window length. The session captures a baseline at construction, so the first
    /// poll returns a (short-window) delta rather than nothing. Returns `nil` only when
    /// unavailable or on a transient sample failure.
    func poll() -> (intervalSeconds: Double, rows: [IOReportChannelSample])? {
        guard let session else { return nil }

        var interval: Double = 0
        guard let array = MacMLXIOReportCopySampleDelta(session, &interval) else {
            return nil
        }

        let count = CFArrayGetCount(array)
        var rows: [IOReportChannelSample] = []
        rows.reserveCapacity(count)
        for index in 0..<count {
            guard let raw = CFArrayGetValueAtIndex(array, index) else { continue }
            let channel = unsafeBitCast(raw, to: CFDictionary.self)
            rows.append(Self.readChannel(channel))
        }
        return (interval, rows)
    }

    // MARK: - Channel marshalling

    private static func readChannel(_ channel: CFDictionary) -> IOReportChannelSample {
        let group = string(MacMLXIOReportChannelGetGroup(channel)) ?? ""
        let subgroup = string(MacMLXIOReportChannelGetSubGroup(channel)) ?? ""
        let name = string(MacMLXIOReportChannelGetChannelName(channel)) ?? ""
        let format = IOReportFormat(rawFormat: MacMLXIOReportChannelGetFormat(channel))

        var simpleValue: Int64?
        var states: [IOReportChannelSample.StateResidency]?
        switch format {
        case .simple:
            simpleValue = MacMLXIOReportSimpleGetIntegerValue(channel)
        case .state:
            let stateCount = MacMLXIOReportStateGetCount(channel)
            if stateCount > 0 {
                var parsed: [IOReportChannelSample.StateResidency] = []
                parsed.reserveCapacity(Int(stateCount))
                for stateIndex in 0..<stateCount {
                    let stateName =
                        string(MacMLXIOReportStateGetNameForIndex(channel, stateIndex)) ?? ""
                    let residency = MacMLXIOReportStateGetResidency(channel, stateIndex)
                    parsed.append(.init(name: stateName, residency: residency))
                }
                states = parsed
            }
        case .invalid, .histogram, .simpleArray, .other:
            break
        }

        return IOReportChannelSample(
            group: group,
            subgroup: subgroup,
            channel: name,
            format: format,
            simpleValue: simpleValue,
            states: states
        )
    }

    /// Bridge a borrowed (+0) CoreFoundation string to a Swift `String`.
    private static func string(_ value: CFString?) -> String? {
        value as String?
    }
}
