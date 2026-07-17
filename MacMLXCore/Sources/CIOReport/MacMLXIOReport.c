//
//  MacMLXIOReport.c
//  CIOReport
//
//  Copyright © 2026 macMLX. English comments only.
//
//  Runtime resolution of the private IOReport symbols plus the session bookkeeping.
//  See MacMLXIOReport.h for why this uses dlopen/dlsym instead of
//  `-undefined dynamic_lookup`, and for the provenance of the signatures below.
//

#include "MacMLXIOReport.h"

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

// IOReport lives in the dyld shared cache; the files do not exist on disk, but
// dlopen resolves these paths against the cache. The location has moved across
// macOS releases — on macOS 26 it is the top-level `/usr/lib/libIOReport.dylib`,
// while older releases exposed it as a PrivateFramework. We try each in turn so a
// single build works across the M1→M5 range macMLX targets. Order is most-current
// first; NULL-terminated.
static const char *const kIOReportCandidatePaths[] = {
    "/usr/lib/libIOReport.dylib",
    "/System/Library/PrivateFrameworks/IOReport.framework/IOReport",
    "/System/Library/PrivateFrameworks/IOReport.framework/Versions/A/IOReport",
    NULL,
};

// Key under which IOReport delta dictionaries carry the per-channel array.
#define kMacMLXIOReportChannelsKey CFSTR("IOReportChannels")

// Private IOReport signatures, mirrored as function pointers.
//
// The trailing uint64_t/CFTypeRef parameters are legacy slots that every known
// caller passes 0/NULL for; they are kept in the signature because the ABI requires
// them, not because we have a use for them.
typedef CFMutableDictionaryRef (*ioreport_copy_all_channels_fn)(uint64_t, uint64_t);
typedef CFMutableDictionaryRef (*ioreport_copy_channels_in_group_fn)(
    CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*ioreport_merge_channels_fn)(
    CFMutableDictionaryRef, CFMutableDictionaryRef, CFTypeRef);
typedef void *(*ioreport_create_subscription_fn)(
    void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*ioreport_create_samples_fn)(
    void *, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*ioreport_create_samples_delta_fn)(
    CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef CFStringRef (*ioreport_channel_get_string_fn)(CFDictionaryRef);
typedef int32_t (*ioreport_channel_get_format_fn)(CFDictionaryRef);
typedef int64_t (*ioreport_simple_get_integer_value_fn)(CFDictionaryRef, int32_t);
typedef int32_t (*ioreport_state_get_count_fn)(CFDictionaryRef);
typedef uint64_t (*ioreport_state_get_residency_fn)(CFDictionaryRef, int32_t);
typedef CFStringRef (*ioreport_state_get_name_for_index_fn)(CFDictionaryRef, int32_t);

static struct {
    ioreport_copy_all_channels_fn copyAllChannels;
    ioreport_copy_channels_in_group_fn copyChannelsInGroup;
    ioreport_merge_channels_fn mergeChannels;
    ioreport_create_subscription_fn createSubscription;
    ioreport_create_samples_fn createSamples;
    ioreport_create_samples_delta_fn createSamplesDelta;
    ioreport_channel_get_string_fn channelGetGroup;
    ioreport_channel_get_string_fn channelGetSubGroup;
    ioreport_channel_get_string_fn channelGetChannelName;
    ioreport_channel_get_format_fn channelGetFormat;
    ioreport_simple_get_integer_value_fn simpleGetIntegerValue;
    ioreport_state_get_count_fn stateGetCount;
    ioreport_state_get_residency_fn stateGetResidency;
    ioreport_state_get_name_for_index_fn stateGetNameForIndex;
    bool available;
    const char *unavailableReason;
} gIOReport;

/// A live subscription plus the rolling state needed to diff consecutive samples.
struct MacMLXIOReportSession {
    void *subscription;                    // private IOReportSubscriptionRef
    CFMutableDictionaryRef subscribedChannels;
    CFDictionaryRef previousSample;        // +1, released on next sample / close
    uint64_t previousTimeNs;               // CLOCK_MONOTONIC_RAW at previousSample
};

/// Monotonic clock in nanoseconds — immune to wall-clock adjustments, so intervals
/// stay correct across NTP steps or sleep.
static uint64_t ktopMonotonicNs(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
}

/// Resolve one symbol, recording the first failure as the unavailable reason.
static void *ktopResolve(void *handle, const char *symbol) {
    void *address = dlsym(handle, symbol);
    if (address == NULL && gIOReport.unavailableReason == NULL) {
        // dlerror()'s buffer is per-thread and transient, so we cannot hand it back
        // as a static string. The symbol name is the part that actually diagnoses.
        gIOReport.unavailableReason = "IOReport symbol missing (see symbol table)";
    }
    return address;
}

static void ktopLoadIOReport(void *context) {
    (void)context;

    void *handle = NULL;
    for (size_t i = 0; kIOReportCandidatePaths[i] != NULL; i++) {
        handle = dlopen(kIOReportCandidatePaths[i], RTLD_LAZY | RTLD_LOCAL);
        if (handle != NULL) {
            break;
        }
    }
    if (handle == NULL) {
        gIOReport.available = false;
        gIOReport.unavailableReason = "IOReport dylib could not be opened";
        return;
    }
    // The handle is intentionally never dlclose()d: the symbols stay live for the
    // process lifetime and unloading a shared-cache image buys nothing.

    gIOReport.copyAllChannels =
        (ioreport_copy_all_channels_fn)ktopResolve(handle, "IOReportCopyAllChannels");
    gIOReport.copyChannelsInGroup =
        (ioreport_copy_channels_in_group_fn)ktopResolve(handle, "IOReportCopyChannelsInGroup");
    gIOReport.mergeChannels =
        (ioreport_merge_channels_fn)ktopResolve(handle, "IOReportMergeChannels");
    gIOReport.createSubscription =
        (ioreport_create_subscription_fn)ktopResolve(handle, "IOReportCreateSubscription");
    gIOReport.createSamples =
        (ioreport_create_samples_fn)ktopResolve(handle, "IOReportCreateSamples");
    gIOReport.createSamplesDelta =
        (ioreport_create_samples_delta_fn)ktopResolve(handle, "IOReportCreateSamplesDelta");
    gIOReport.channelGetGroup =
        (ioreport_channel_get_string_fn)ktopResolve(handle, "IOReportChannelGetGroup");
    gIOReport.channelGetSubGroup =
        (ioreport_channel_get_string_fn)ktopResolve(handle, "IOReportChannelGetSubGroup");
    gIOReport.channelGetChannelName =
        (ioreport_channel_get_string_fn)ktopResolve(handle, "IOReportChannelGetChannelName");
    gIOReport.channelGetFormat =
        (ioreport_channel_get_format_fn)ktopResolve(handle, "IOReportChannelGetFormat");
    gIOReport.simpleGetIntegerValue =
        (ioreport_simple_get_integer_value_fn)ktopResolve(handle, "IOReportSimpleGetIntegerValue");
    gIOReport.stateGetCount =
        (ioreport_state_get_count_fn)ktopResolve(handle, "IOReportStateGetCount");
    gIOReport.stateGetResidency =
        (ioreport_state_get_residency_fn)ktopResolve(handle, "IOReportStateGetResidency");
    gIOReport.stateGetNameForIndex =
        (ioreport_state_get_name_for_index_fn)ktopResolve(handle, "IOReportStateGetNameForIndex");

    gIOReport.available = gIOReport.copyAllChannels != NULL &&
                          gIOReport.copyChannelsInGroup != NULL &&
                          gIOReport.mergeChannels != NULL &&
                          gIOReport.createSubscription != NULL &&
                          gIOReport.createSamples != NULL &&
                          gIOReport.createSamplesDelta != NULL &&
                          gIOReport.channelGetGroup != NULL &&
                          gIOReport.channelGetSubGroup != NULL &&
                          gIOReport.channelGetChannelName != NULL &&
                          gIOReport.channelGetFormat != NULL &&
                          gIOReport.simpleGetIntegerValue != NULL &&
                          gIOReport.stateGetCount != NULL &&
                          gIOReport.stateGetResidency != NULL &&
                          gIOReport.stateGetNameForIndex != NULL;

    if (gIOReport.available) {
        gIOReport.unavailableReason = NULL;
    }
}

/// Resolve the symbol table exactly once, whichever thread arrives first.
static void ktopEnsureLoaded(void) {
    static dispatch_once_t onceToken;
    dispatch_once_f(&onceToken, NULL, ktopLoadIOReport);
}

// MARK: - Availability

bool MacMLXIOReportIsAvailable(void) {
    ktopEnsureLoaded();
    return gIOReport.available;
}

const char *MacMLXIOReportUnavailableReason(void) {
    ktopEnsureLoaded();
    return gIOReport.available ? NULL : gIOReport.unavailableReason;
}

// MARK: - Channel discovery

CFMutableDictionaryRef MacMLXIOReportCopyAllChannels(void) {
    if (!MacMLXIOReportIsAvailable()) {
        return NULL;
    }
    return gIOReport.copyAllChannels(0, 0);
}

CFMutableDictionaryRef MacMLXIOReportCopyChannelsInGroup(
    CFStringRef group, CFStringRef subgroup) {
    if (!MacMLXIOReportIsAvailable()) {
        return NULL;
    }
    return gIOReport.copyChannelsInGroup(group, subgroup, 0, 0, 0);
}

bool MacMLXIOReportMergeChannels(
    CFMutableDictionaryRef destination, CFMutableDictionaryRef source) {
    if (!MacMLXIOReportIsAvailable()) {
        return false;
    }
    gIOReport.mergeChannels(destination, source, NULL);
    return true;
}

// MARK: - Session lifecycle

MacMLXIOReportSessionRef MacMLXIOReportOpen(CFMutableDictionaryRef desiredChannels) {
    if (!MacMLXIOReportIsAvailable() || desiredChannels == NULL) {
        return NULL;
    }

    CFMutableDictionaryRef subscribed = NULL;
    void *subscription =
        gIOReport.createSubscription(NULL, desiredChannels, &subscribed, 0, NULL);
    if (subscription == NULL) {
        if (subscribed != NULL) {
            CFRelease(subscribed);
        }
        return NULL;
    }

    struct MacMLXIOReportSession *session = calloc(1, sizeof(*session));
    if (session == NULL) {
        CFRelease((CFTypeRef)subscription);
        if (subscribed != NULL) {
            CFRelease(subscribed);
        }
        return NULL;
    }

    session->subscription = subscription;
    session->subscribedChannels = subscribed;  // may be NULL; sampling still works
    session->previousSample = gIOReport.createSamples(subscription, subscribed, NULL);
    session->previousTimeNs = ktopMonotonicNs();
    return session;
}

void MacMLXIOReportClose(MacMLXIOReportSessionRef session) {
    if (session == NULL) {
        return;
    }
    if (session->previousSample != NULL) {
        CFRelease(session->previousSample);
    }
    if (session->subscribedChannels != NULL) {
        CFRelease(session->subscribedChannels);
    }
    if (session->subscription != NULL) {
        CFRelease((CFTypeRef)session->subscription);
    }
    free(session);
}

CFArrayRef MacMLXIOReportCopySampleDelta(
    MacMLXIOReportSessionRef session, double *outIntervalSeconds) {
    if (outIntervalSeconds != NULL) {
        *outIntervalSeconds = 0.0;
    }
    if (!MacMLXIOReportIsAvailable() || session == NULL) {
        return NULL;
    }

    CFDictionaryRef current =
        gIOReport.createSamples(session->subscription, session->subscribedChannels, NULL);
    if (current == NULL) {
        return NULL;
    }
    uint64_t currentTimeNs = ktopMonotonicNs();

    CFArrayRef channels = NULL;
    if (session->previousSample != NULL) {
        CFDictionaryRef delta =
            gIOReport.createSamplesDelta(session->previousSample, current, NULL);
        if (delta != NULL) {
            CFArrayRef inner = CFDictionaryGetValue(delta, kMacMLXIOReportChannelsKey);
            if (inner != NULL) {
                channels = (CFArrayRef)CFRetain(inner);  // outlive the delta dict
            }
            CFRelease(delta);
        }
        if (outIntervalSeconds != NULL) {
            *outIntervalSeconds =
                (double)(currentTimeNs - session->previousTimeNs) / 1.0e9;
        }
    }

    // Advance the rolling baseline to the sample we just took.
    if (session->previousSample != NULL) {
        CFRelease(session->previousSample);
    }
    session->previousSample = current;  // takes ownership of the +1 from createSamples
    session->previousTimeNs = currentTimeNs;

    return channels;
}

// MARK: - Channel accessors

CFStringRef MacMLXIOReportChannelGetGroup(CFDictionaryRef channel) {
    if (!MacMLXIOReportIsAvailable()) {
        return NULL;
    }
    return gIOReport.channelGetGroup(channel);
}

CFStringRef MacMLXIOReportChannelGetSubGroup(CFDictionaryRef channel) {
    if (!MacMLXIOReportIsAvailable()) {
        return NULL;
    }
    return gIOReport.channelGetSubGroup(channel);
}

CFStringRef MacMLXIOReportChannelGetChannelName(CFDictionaryRef channel) {
    if (!MacMLXIOReportIsAvailable()) {
        return NULL;
    }
    return gIOReport.channelGetChannelName(channel);
}

int32_t MacMLXIOReportChannelGetFormat(CFDictionaryRef channel) {
    if (!MacMLXIOReportIsAvailable()) {
        return MacMLXIOReportFormatInvalid;
    }
    return gIOReport.channelGetFormat(channel);
}

int64_t MacMLXIOReportSimpleGetIntegerValue(CFDictionaryRef channel) {
    if (!MacMLXIOReportIsAvailable()) {
        return 0;
    }
    // The second parameter is an unused legacy slot in every known caller.
    return gIOReport.simpleGetIntegerValue(channel, 0);
}

int32_t MacMLXIOReportStateGetCount(CFDictionaryRef channel) {
    if (!MacMLXIOReportIsAvailable()) {
        return 0;
    }
    return gIOReport.stateGetCount(channel);
}

uint64_t MacMLXIOReportStateGetResidency(CFDictionaryRef channel, int32_t index) {
    if (!MacMLXIOReportIsAvailable()) {
        return 0;
    }
    return gIOReport.stateGetResidency(channel, index);
}

CFStringRef MacMLXIOReportStateGetNameForIndex(CFDictionaryRef channel, int32_t index) {
    if (!MacMLXIOReportIsAvailable()) {
        return NULL;
    }
    return gIOReport.stateGetNameForIndex(channel, index);
}
