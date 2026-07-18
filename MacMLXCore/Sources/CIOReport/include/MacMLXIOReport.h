//
//  MacMLXIOReport.h
//  CIOReport
//
//  Copyright © 2026 macMLX. English comments only.
//
//  Runtime bridge to Apple's private IOReport framework.
//
//  WHY THIS EXISTS
//  ---------------
//  Apple ships no public API for GPU residency, per-requestor DRAM bandwidth, or
//  ANE power. Every Apple Silicon monitoring tool reads them through IOReport, a
//  private framework with no headers in the SDK. This target declares the handful
//  of symbols we need and resolves them at runtime.
//
//  WHY dlopen/dlsym RATHER THAN `-undefined dynamic_lookup`
//  --------------------------------------------------------
//  The usual trick is to declare the symbols `extern` and pass
//  `-undefined dynamic_lookup` to the linker. We deliberately do not:
//
//   1. MacMLXCore is a *library* consumed by two independent front ends (the
//      SwiftUI app via macMLX.xcodeproj, the CLI via SPM) plus a test target.
//      `-undefined dynamic_lookup` must be applied to every *final binary*, so
//      each consumer — and both CI lanes — would have to carry the flag. New
//      consumers would fail at link time in a way that is not obviously our fault.
//   2. The flag disables undefined-symbol checking for the *entire* binary, not
//      just IOReport. A typo in any unrelated symbol degrades from a link error
//      into a runtime crash. That is a bad trade for a library others link.
//   3. dlopen/dlsym degrades gracefully: if a symbol cannot be resolved we report
//      "unavailable" and the callers return nil. If a future macOS renames or
//      removes IOReport, macMLX keeps launching and simply stops showing the
//      silicon panel.
//
//  SYMBOL PROVENANCE / LICENSING
//  -----------------------------
//  These declarations were written for macMLX from publicly documented signatures
//  (the reverse-engineering lineage runs through NeoAsitop, macmon, and
//  kennss/SiliconScope, all MIT). Function *signatures* are interface facts, not
//  copyrightable expression, and no implementation is copied from any of those
//  projects — the dlopen resolver, the session model, and the API shape below are
//  our own. Listed purely as an attribution courtesy to the prior art.
//
//  API SHAPE
//  ---------
//  Swift never touches the raw private symbols or the subscription-ownership dance.
//  It:
//    1. builds a "desired channels" dictionary with
//       `MacMLXIOReportCopyChannelsInGroup` + `MacMLXIOReportMergeChannels`,
//    2. opens a session over it (`MacMLXIOReportOpen`), and
//    3. polls `MacMLXIOReportCopySampleDelta`, which internally diffs against the
//       previous sample and returns the per-channel deltas plus the elapsed wall
//       time. The session owns the subscription, the rolling previous sample, and
//       the monotonic clock — the messy CoreFoundation lifetimes stay in C.
//  Each returned channel is a CoreFoundation dictionary read back through the
//  `...ChannelGet...` / `...StateGet...` accessors.
//
//  OWNERSHIP CONTRACT
//  ------------------
//  Standard CoreFoundation naming rules, so Swift's implicit bridging manages
//  lifetimes correctly:
//    * `...Copy...` returns +1 — the caller (Swift) owns and auto-releases it.
//    * `...Get...`  returns +0 — borrowed, do not release.
//  The session handle is opaque and must be freed once with `MacMLXIOReportClose`.
//
//  THREAD SAFETY
//  -------------
//  Symbol resolution is one-shot and thread safe. A single session must NOT be
//  polled concurrently from two threads — the Swift owner (an actor) serialises it.
//

#ifndef MACMLX_IOREPORT_H
#define MACMLX_IOREPORT_H

#include <CoreFoundation/CoreFoundation.h>
#include <stdbool.h>
#include <stdint.h>

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

/// Opaque handle to an open sampling session (subscription + rolling state).
///
/// Deliberately not a CF type: we never want ARC or Swift's implicit CF bridging
/// guessing at the lifetime of a handle whose contents are private. Free it exactly
/// once with `MacMLXIOReportClose`.
typedef struct MacMLXIOReportSession *MacMLXIOReportSessionRef;

/// Channel value encodings reported by `MacMLXIOReportChannelGetFormat`.
///
/// Only `Simple` (a monotonic counter) and `State` (per-state residency ticks) are
/// consumed by macMLX today; the others are declared so an unexpected format can be
/// surfaced honestly rather than silently misread as a counter.
typedef CF_ENUM(int32_t, MacMLXIOReportFormat) {
    MacMLXIOReportFormatInvalid = 0,
    MacMLXIOReportFormatSimple = 1,
    MacMLXIOReportFormatState = 2,
    MacMLXIOReportFormatHistogram = 3,
    MacMLXIOReportFormatSimpleArray = 4,
};

// MARK: - Availability

/// True when every IOReport symbol macMLX needs resolved successfully.
///
/// Resolution happens once, on first call, and is cached. When this returns false
/// every other function here is a safe no-op returning NULL/0 — callers should treat
/// silicon metrics as unavailable rather than failing.
bool MacMLXIOReportIsAvailable(void);

/// Human-readable reason `MacMLXIOReportIsAvailable()` returned false.
///
/// Returns NULL when IOReport is available. The string is a static constant and
/// must not be freed. Intended for diagnostics and test skip messages.
const char *_Nullable MacMLXIOReportUnavailableReason(void);

// MARK: - Channel discovery

/// Copy the channel set for `group` (optionally narrowed to `subgroup`).
///
/// Pass NULL for `subgroup` to take every subgroup in the group. Returns NULL when
/// IOReport is unavailable or the group does not exist on this chip — channel layout
/// is chip- and OS-specific, so a NULL here is normal, not an error.
CF_RETURNS_RETAINED
CFMutableDictionaryRef _Nullable MacMLXIOReportCopyChannelsInGroup(
    CFStringRef group,
    CFStringRef _Nullable subgroup);

/// Copy the full channel set advertised by this machine.
///
/// Used by the channel-discovery probe to record what actually exists on a given
/// chip. Production samplers should ask for the specific groups they need instead.
CF_RETURNS_RETAINED
CFMutableDictionaryRef _Nullable MacMLXIOReportCopyAllChannels(void);

/// Merge `source`'s channels into `destination`, so one session can span several
/// groups. Returns false when IOReport is unavailable.
bool MacMLXIOReportMergeChannels(
    CFMutableDictionaryRef destination,
    CFMutableDictionaryRef source);

// MARK: - Session lifecycle

/// Open a sampling session over `desiredChannels` and capture a baseline sample.
///
/// Returns NULL when IOReport is unavailable or the subscription is rejected. The
/// returned session owns an internal subscription and a rolling "previous sample";
/// the first `MacMLXIOReportCopySampleDelta` diffs against this baseline.
MacMLXIOReportSessionRef _Nullable MacMLXIOReportOpen(
    CFMutableDictionaryRef desiredChannels);

/// Close a session opened by `MacMLXIOReportOpen`. Tolerates NULL.
void MacMLXIOReportClose(MacMLXIOReportSessionRef _Nullable session);

/// Sample now, diff against the session's previous sample, and return the deltas.
///
/// The result is the CFArray of per-channel dictionaries (each read via the
/// accessors below); values are the change accumulated since the previous call.
/// `outIntervalSeconds`, when non-NULL, receives the monotonic wall time between the
/// two samples — the denominator for turning byte/energy deltas into rates. Returns
/// NULL on failure (leaving `*outIntervalSeconds` at 0). The session advances its
/// previous sample to the one just taken.
CF_RETURNS_RETAINED
CFArrayRef _Nullable MacMLXIOReportCopySampleDelta(
    MacMLXIOReportSessionRef session,
    double *_Nullable outIntervalSeconds);

// MARK: - Channel accessors (read a channel dictionary from a delta array)

/// Group name of a channel dictionary (borrowed, +0).
CFStringRef _Nullable MacMLXIOReportChannelGetGroup(CFDictionaryRef channel);

/// Subgroup name of a channel dictionary (borrowed, +0). NULL when ungrouped.
CFStringRef _Nullable MacMLXIOReportChannelGetSubGroup(CFDictionaryRef channel);

/// Channel name of a channel dictionary (borrowed, +0).
CFStringRef _Nullable MacMLXIOReportChannelGetChannelName(CFDictionaryRef channel);

/// Value encoding of a channel dictionary. See `MacMLXIOReportFormat`.
int32_t MacMLXIOReportChannelGetFormat(CFDictionaryRef channel);

/// Scalar value of a `Simple`-format channel.
///
/// In a delta dictionary this is the counter's increase over the interval — for the
/// Energy Model group that is millijoules, for AMC performance counters it is bytes.
int64_t MacMLXIOReportSimpleGetIntegerValue(CFDictionaryRef channel);

/// Number of states in a `State`-format channel.
int32_t MacMLXIOReportStateGetCount(CFDictionaryRef channel);

/// Residency ticks accumulated in state `index`.
///
/// In a delta dictionary these are the ticks spent in that state during the
/// interval; a state's share of the row's total is its fraction of the interval.
uint64_t MacMLXIOReportStateGetResidency(CFDictionaryRef channel, int32_t index);

/// Name of state `index` (borrowed, +0), e.g. `"OFF"`, `"P1"`, `"32GB/s"`.
CFStringRef _Nullable MacMLXIOReportStateGetNameForIndex(CFDictionaryRef channel, int32_t index);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

#endif /* MACMLX_IOREPORT_H */
