#!/usr/bin/env python3
"""Hold a root's SwiftPM lockfile to MacMLXCore's.

Three roots build shippable artifacts (MacMLXCore's tests, the macmlx CLI, the
app's DMG), and each resolves its own dependency graph. MacMLXCore's lockfile is
the graph MacMLXCore's tests ran on; this script fails when another root's
lockfile pins any package MacMLXCore also pins at a different version or
revision, or leaves one of them out. Before it existed the app had no committed
lockfile at all, and the v0.9.0 DMG shipped swift-jinja 2.4.2 while every
chat-template test had run on 2.3.6 (#149).

A package only MacMLXCore's tests depend on never reaches another root's
graph (SwiftPM ignores a dependency's test-only requirements when another
root resolves), so it would be reported missing forever; name it in
CORE_TEST_ONLY when that day comes. Today every such package also arrives
through a library dependency, so the set is empty.

Usage: check-lockfile-parity.py [--github] <lockfile> [<reference lockfile>]

`--github` prefixes failures with `::error::` for GitHub Actions annotations.
The reference defaults to MacMLXCore/Package.resolved. Exit status 1 on any
difference, 2 when a lockfile cannot be read or the reference pins nothing.
"""

import json
import sys

FORK = "https://github.com/magicnight/mlx-swift.git"
CORE_TEST_ONLY = frozenset()


def normalized(location):
    """One spelling for a package URL: lockfiles differ in case and `.git`."""
    return location.strip().lower().rstrip("/").removesuffix(".git")


def pins(path):
    """identity -> (location, version, revision) for every pin in a lockfile."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    entries = document.get("pins") or document.get("object", {}).get("pins", [])
    result = {}
    for pin in entries:
        state = pin.get("state", {})
        location = pin.get("location") or pin.get("repositoryURL", "")
        identity = pin.get("identity") or location.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git").lower()
        result[identity] = (location, state.get("version"), state.get("revision"))
    return result


def main(argv):
    prefix = ""
    if argv and argv[0] == "--github":
        prefix = "::error::"
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2
    lockfile = argv[0]
    reference = argv[1] if len(argv) > 1 else "MacMLXCore/Package.resolved"
    try:
        checked = pins(lockfile)
        expected = pins(reference)
    except (OSError, ValueError, AttributeError, TypeError, KeyError) as error:
        print(f"{prefix}cannot read a lockfile: {error}")
        return 2
    if not expected:
        print(f"{prefix}{reference} pins nothing; there is no graph to hold {lockfile} to.")
        return 2

    failures = []
    for identity, (location, version, revision) in sorted(expected.items()):
        actual = checked.get(identity)
        if actual is None:
            if identity in CORE_TEST_ONLY:
                continue
            failures.append(f"{identity}: {reference} pins it ({version or revision}) and {lockfile} does not name it")
            continue
        actual_location, actual_version, actual_revision = actual
        if normalized(actual_location) != normalized(location):
            hint = ""
            if identity == "mlx-swift" and normalized(actual_location) != normalized(FORK):
                hint = " (the controlled fork's cherry-picks would not ship)"
            failures.append(f"{identity}: {lockfile} resolved {actual_location}, {reference} {location}{hint}")
        if actual_revision != revision or actual_version != version:
            failures.append(
                f"{identity}: {lockfile} pins {actual_version or ''} {actual_revision or ''}, "
                f"{reference} pins {version or ''} {revision or ''}"
            )
    for failure in failures:
        print(f"{prefix}{failure}")
    if failures:
        print(f"{prefix}{lockfile} differs from {reference} in {len(failures)} place(s); the artifact built from it would ship a graph the tests never ran on.")
        return 1
    extra = sorted(set(checked) - set(expected))
    fork = checked.get("mlx-swift")
    fork_note = f" (mlx-swift at {fork[0]} {(fork[2] or '')[:7]})" if fork else ""
    print(f"{lockfile}: every one of the {len(expected)} packages {reference} pins is pinned the same way{fork_note}"
          + (f"; this root alone pins {', '.join(extra)}" if extra else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
