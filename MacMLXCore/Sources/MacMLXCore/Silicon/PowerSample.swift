// Copyright © 2026 macMLX. English comments only.

/// Average power draw for one Energy Model domain over a window, in watts.
///
/// The `Energy Model` group exposes each domain (`CPU Energy`, `GPU`, `ANE`, `DRAM`)
/// as a `Simple` energy counter in millijoules. Over a window the delta is energy
/// consumed, and dividing by the elapsed seconds gives average power.
///
/// For the ANE domain this is the ONLY signal available — there is no ANE occupancy
/// or utilisation counter. `SiliconSample.anePower` therefore carries a power *proxy*
/// only, and callers must not present it as an ANE utilisation percentage.
public struct PowerSample: Sendable, Equatable {

    /// Average power over the window, in watts.
    public let watts: Double

    public init(watts: Double) {
        self.watts = watts
    }

    /// Convert an energy delta (millijoules) over `intervalSeconds` to average watts.
    ///
    /// Returns `nil` when the interval is non-positive (e.g. the first sample, before a
    /// window exists). A negative energy delta — which should not occur but could on a
    /// counter reset — is clamped to zero rather than reported as negative power.
    public static func from(energyMillijoules: Int64, intervalSeconds: Double) -> PowerSample? {
        guard intervalSeconds > 0 else { return nil }
        let joules = Double(max(0, energyMillijoules)) / 1000.0
        return PowerSample(watts: joules / intervalSeconds)
    }
}
