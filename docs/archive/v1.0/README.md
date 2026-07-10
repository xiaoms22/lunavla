# v1.0 historical archive

These files are preserved as a correction-before snapshot of the initial LunaVLA teaching release. They are useful for provenance, but they are not current documentation or publishable v1.1 evidence.

Important corrections:

- The v1.0 linear NumPy chunk predictor was described as ACT in several reports; it is not an Action Chunking with Transformers implementation.
- The synthetic state-only point-reach task was sometimes described as PushT or VLA; it has no images, T-block physics, or real-robot interface.
- Reported BC-to-chunk differences changed more than one experimental factor and used too few evaluation episodes to establish a causal chunking effect.
- Paths under `outputs/` and absolute workstation paths described transient local artifacts that are not present in the public source tree.
- Diffusion Policy media was an external reference, not output produced by this repository.

Current facts live in the root `README.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `ROADMAP.md`, and `CHANGELOG.md`. New results must be generated from v1.1 manifests under `results/v1.1/`.

Archive layout:

- root Markdown files: completion, release, delivery, and migration narratives;
- `docs/`: the former internship pack, report templates, tutorials, and portfolio guidance;
- `scripts/`: unsupported report/media generators and ad hoc v1.0 experiment orchestration;
- `media/`: pre-correction SVG result cards, external ACT/Diffusion reference GIFs, and their old manifests.

Archived scripts preserve their original relative-path assumptions and are not expected to run from this directory. They must not be linked as current commands.
