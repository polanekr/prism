# Repository transition — 2026-11-16

The default `main` branch now contains the PRISM desktop application from the maintainer's `prism-production` worktree. The imported source baseline is `b8c7a1e957e978d28bc84a6753577bf3f884b869`; the scientific application code is unchanged by this repository transition.

The previous GitHub framework is retained at commit `a00c9e1fa7b07ab1a571f26d262ee5019179ea51` and on the `legacy/framework-2026-11-16` branch. Its original commits are also ancestors of the transition merge commit. No force push or repository deletion is needed.

The two projects had unrelated Git roots. The transition deliberately retains the current desktop tree while joining both histories; it is not a source-code combination of the two architectures. The existing MIT licence notice is retained. Earlier datasets, generated figures, notebooks and package-build artifacts remain in legacy history rather than the current desktop source tree.

## Scope and versioning

- `v0.1.0`: the current desktop application's first GitHub source release tag.
- Legacy README references to framework v1.0, Gaussian-process volume reconstruction and biological models do not describe this desktop release.
- Existing calibration and project files must be checked for compatibility; do not assume that the framework API, imports or examples apply to the desktop application.
- The repository tag identifies a source snapshot. It does not assert scanner validation, Windows/Linux CI success or clinical suitability.

## Documentation

English documentation: https://polanek.hu/en/software/prism/

During the initial website upload, this documentation URL may not yet be live. Installation and current implementation notes are also available in `README.md`.
