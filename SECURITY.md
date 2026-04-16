# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | ✅        |

## Reporting a Vulnerability

macMLX runs entirely on-device with no network services exposed beyond
`localhost`. However, if you discover a security issue:

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report it via [GitHub Private Vulnerability Reporting](../../security/advisories/new).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix if you have one

You can expect an acknowledgement within 72 hours and a resolution timeline
within 14 days for confirmed issues.

## Scope

- The macMLX Swift application
- The Python inference backend (`Backend/`)
- The local HTTP API (localhost only)

## Out of Scope

- Vulnerabilities in upstream dependencies (mlx-lm, FastAPI, etc.)
  — please report those to their respective projects
- Issues requiring physical access to the machine
