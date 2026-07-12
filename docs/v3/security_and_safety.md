# v3 security, privacy and safety boundary

- Release operations require a clean immutable SHA, strict artifact containment, signed tag,
  required checks, SBOM, provenance and exact checksums.
- Self-hosted GPU jobs use ephemeral isolated runners with one visible A100 and no private mounts.
  They upload only small manifests and hash inventories.
- Unverified weight licenses, dirty evidence, mixed dependencies, missing pairs, non-finite values,
  hash drift and private-data leakage all fail closed.
- Public documents and assets are scanned for credentials, private paths, internal URLs, people,
  company implementation details and unpublished metrics.
- `deploy` parity means checkpoint preprocessing parity only; it is not a physical-robot deployment
  claim.

Out of scope are production services, physical robot deployment, ROS2/WBC/SLAM stacks, multi-node
training, safety certification and benchmark leadership claims.

