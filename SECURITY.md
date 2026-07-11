# Security Policy

## Supported versions

Security fixes are provided for the latest stable release on `main` and for the
maintained NumPy teaching line on `v1.x`. Historical tags and archived evidence
snapshots remain available for reproducibility but are not separate maintenance
branches.

The `v2.0.0-rc.1` line on protected branch `v2` receives pre-release security
fixes until v2.0.0 reaches `main`. Its CPU-only and non-robot support boundary
does not change during the release-candidate period.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, exposed credential,
private dataset, or identifying rollout artifact. Use GitHub's private
vulnerability reporting feature for this repository. Include the affected
version or commit, reproduction steps, impact, and any proposed mitigation.

Maintainers will acknowledge a report within seven days, validate its scope, and
coordinate disclosure. Never include live credentials in a report; revoke them
first and replace them with redacted examples.

## Project boundary

LunaVLA is a CPU-first educational simulation. It is not intended to control a
physical robot, enforce operational safety constraints, or protect production
secrets. Reports about unsupported real-robot deployment should be directed to
the system owner rather than treated as guarantees of this project.
