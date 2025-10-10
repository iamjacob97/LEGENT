# hazalyser

hazalyser provides the hazard analysis layer for LEGENT. It builds reproducible indoor scenes, collects multimodal observations, and runs Environmental Survey Hazard Analysis (ESHA) workflows backed by large language models so researchers can catalogue and explain assistive-robotics risks.

## Core Responsibilities
- Generate single-room scenes with tunable clutter, spacing, and subject placement aimed at surfacing hazards.
- Capture top-down, side, and egocentric views that ground downstream LLM reasoning.
- Load ESHA framework assets and assemble prompts that yield structured hazard narratives and mitigation notes.
- Persist scene metadata, analysis outputs, and supporting artefacts alongside LEGENT for experiment reproducibility.

## Development Guidelines
- Follow PEP 8 with type annotations and keep comments for non-obvious logic.
- Store new framework prompts under `frameworks/` and document how they modify ESHA flows.
- Add or update regression tests when you change controller commands, generator behaviour, or prompt construction.
- Prefer modular helpers over duplicating LEGENT functionality so the hazard layer stays maintainable.

## Further Reading
Consult the design artefacts and complete user manual under `docs/` for extended workflows, evaluation metrics, and integration practices tied to the hazard analysis pipeline.
