# Security Policy

## Supported versions

Security fixes are provided for the latest stable release on `main` and for the
maintained NumPy teaching line on `v1.x`. Historical tags and archived evidence
snapshots remain available for reproducibility but are not separate maintenance
branches.

The `v2.0.0` line on protected default branch `main` receives security fixes.
Its supported evidence boundary is CPU Linux; experimental CUDA workflows are
not a supported security boundary. The `v1.x` maintenance branch continues to
receive NumPy teaching-core fixes.

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
