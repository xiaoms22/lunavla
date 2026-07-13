# v3 security, privacy and safety boundary

Stable PyPI publication uses no long-lived API token. The dedicated workflow receives job-scoped
OIDC permission only inside the protected `pypi` Environment, accepts only the verified annotated
`v3.0.0` tag, and uploads the signed evidence distributions without rebuilding them. Publication
provenance and file digests are verified before GitHub marks the release stable.

- Release operations require a clean immutable SHA, strict artifact containment, signed tag,
  required checks, SBOM, provenance and exact checksums.
- v3.0 release authority is hosted Linux CPU. Experimental self-hosted GPU validation remains a
  separate v3.1 gate and cannot affect v3.0 claims.
- Unverified weight licenses, dirty evidence, mixed dependencies, missing pairs, non-finite values,
  hash drift and private-data leakage all fail closed.
- Public documents and assets are scanned for credentials, private paths, internal URLs, people,
  company implementation details and unpublished metrics.
- `deploy` parity means checkpoint preprocessing parity only; it is not a physical-robot deployment
  claim.

Out of scope are production services, physical robot deployment, ROS2/WBC/SLAM stacks, multi-node
training, safety certification and benchmark leadership claims.
