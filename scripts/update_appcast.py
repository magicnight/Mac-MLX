#!/usr/bin/env python3
"""scripts/update_appcast.py — Prepend a new <item> to appcast.xml on each release.

Invoked by .github/workflows/release.yml after the DMG is signed by Sparkle's
sign_update tool. The appcast.xml is committed back to main so Sparkle clients
discover the new version.

Usage:
    update_appcast.py --version 0.1.0 \\
                      --build 42 \\
                      --signature "<base64 EdDSA from sign_update>" \\
                      --size 12345678
"""

from __future__ import annotations

import argparse
import datetime
import sys
import xml.etree.ElementTree as ET

SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"
DEFAULT_REPO = "magicnight/Mac-MLX"
DEFAULT_APPCAST = "appcast.xml"
DEFAULT_MIN_MACOS = "14.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Semver, e.g. 0.1.0")
    parser.add_argument("--build", required=True, type=int, help="Integer build number (GITHUB_RUN_NUMBER)")
    parser.add_argument("--signature", required=True, help="EdDSA signature from sign_update")
    parser.add_argument("--size", required=True, type=int, help="DMG file size in bytes")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo (case-sensitive in raw URLs)")
    parser.add_argument("--appcast", default=DEFAULT_APPCAST, help="Path to appcast.xml")
    parser.add_argument("--min-macos", default=DEFAULT_MIN_MACOS, help="Minimum macOS version")
    args = parser.parse_args()

    ET.register_namespace("sparkle", SPARKLE_NS)
    tree = ET.parse(args.appcast)
    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        print("error: malformed appcast.xml (no <channel>)", file=sys.stderr)
        return 1

    item = ET.Element("item")
    ET.SubElement(item, "title").text = f"Version {args.version}"
    ET.SubElement(item, f"{{{SPARKLE_NS}}}version").text = str(args.build)
    ET.SubElement(item, f"{{{SPARKLE_NS}}}shortVersionString").text = args.version
    ET.SubElement(item, f"{{{SPARKLE_NS}}}minimumSystemVersion").text = args.min_macos
    ET.SubElement(item, "pubDate").text = datetime.datetime.now(datetime.UTC).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )

    enclosure = ET.SubElement(item, "enclosure")
    enclosure.set(
        "url",
        f"https://github.com/{args.repo}/releases/download/v{args.version}/macMLX-v{args.version}.dmg",
    )
    enclosure.set(f"{{{SPARKLE_NS}}}edSignature", args.signature)
    enclosure.set("length", str(args.size))
    enclosure.set("type", "application/octet-stream")

    ET.SubElement(item, f"{{{SPARKLE_NS}}}releaseNotesLink").text = (
        f"https://github.com/{args.repo}/releases/tag/v{args.version}"
    )

    # Insert as first <item> after the channel meta. Sparkle clients prefer
    # the most recent at the top.
    insert_idx = 0
    for i, child in enumerate(channel):
        if child.tag == "item":
            insert_idx = i
            break
        insert_idx = i + 1
    channel.insert(insert_idx, item)

    ET.indent(tree, space="  ")
    tree.write(args.appcast, encoding="utf-8", xml_declaration=True)
    print(f"appcast.xml updated: v{args.version} (build {args.build}, {args.size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
