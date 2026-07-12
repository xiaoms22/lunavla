# CPU policy profiling

`PolicyProfileDesignV1` fixes five warmup operations and twenty measured operations on Python 3.12
CPU. The design records batch size, thread count, accepted operating systems, base config and output
directory. `profile-run` records training and inference latency, sample throughput, peak RSS, the
resolved config, the exact dependency lock and a privacy-safe environment description.

```bash
lunavla-v3 profile-run configs/v3/profile_numpy_cpu.yaml
lunavla-v3 profile-verify outputs/v3/profiles/numpy-linear-chunk
```

`PolicyProfileManifestV1` hashes every source artifact. Verification recomputes the latency summaries
from all twenty measurements and rejects missing, extra or modified files. Dirty runs remain
inspectable but are not release eligible.

Profiles from different machines are reported separately. They do not establish that ACT, Diffusion
or another policy is faster or better, and the fixed manifest wording always keeps
`comparative_claim_allowed=false`. MPS may be used for development smoke only; the v3.0 authority
remains CPU Linux.
