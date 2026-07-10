# Security Policy

## Supported versions

Security fixes are provided for the latest release on the `main` branch. Older
teaching snapshots may be retained for reproducibility but are not maintained.

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
