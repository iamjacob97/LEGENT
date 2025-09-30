# hazalyser

## Overview
hazalyser provides a simulation-driven hazard analysis layer on top of LEGENT. It couples procedural environment generation with  frameworks like Environmental Survey Hazard Analysis (ESHA) prompts so that assistive-robotics risks can be explored, documented, and benchmarked in reproducible single-room scenarios. It is extremely modular and extensible with method analysis on the simulated scenes like FMEA and HAZOP.

Core capabilities include:
- Single-room scene synthesis with controllable clutter, spacing, and asset overrides.
- Automated capture of top-down and side perspectives for visual inspection.
- LLM-backed ESHA assessments that turn spatial context into structured hazard narratives.
- Utilities for loading framework prompts, subject meshes, and environment assets co-located with the LEGENT package.

## Module Layout
- `controller.py` – chat-oriented runtime that drives scene creation, adjustment, and ESHA analysis cycles.
- `generator.py` – custom `SceneGenerator` built for hazard studies; wraps LEGENT placement routines and integrates subject/agent logic.
- `helpers.py` – shared helpers for scene state introspection, prompt loading, mesh sizing, and env var management.
- `house.py` – single-room structural primitives (`HazardRoom`, `HazardHouse`) and generation helpers.
- `objects.py` – wrapper around LEGENT asset metadata with lazy-loading convenience utilities.
- `prompter.py` – constructs LLM requests, parses scene state, and records analysis outputs.
- `smallObjects.py`, `obsAssets/`, `frameworks/`, `llm_analysis/` – supporting assets for object placement, perception, and saved LLM results.
- `tests/` – regression and scenario tests for generator correctness and prompt handling.

Additional design references live in `docs/`:
- `hazalyser_module_uml.puml` – class/interaction diagram for the hazard analysis stack.
- `scene_generation_high_level_design.md` – narrative description of the custom scene pipeline.
- `scene_generation_pipeline.puml` – activity diagram for the generator phases.

## Installation
1. Install LEGENT and its dependencies (Python 3.10 recommended) following the root project instructions.
2. Ensure `openai` is available and updated if you plan to run ESHA analyses (`pip install openai`).
3. Export any required LLM credentials as JSON strings in your environment variables (referenced via `helpers.get_env_key`).

## Quick Start
```python
from hazalyser.controller import Controller

config = SceneConfig(framework="esha", llm_key="openai_default")
controller = Controller(scene_config=config)
controller.start(env_path="auto")
```

Within the LEGENT interface, issue commands such as `#NEW`, `#CLUTTER+`, `#LOCK`, and `#FWANALYSE` to iterate on scenes and trigger hazard evaluations.

### Batch Generation Only
If you only need the procedural scene data:
```python
from hazalyser.generator import SceneGenerator, SceneConfig

generator = SceneGenerator(scene_config=SceneConfig(framework="esha"))
scene_bundle = generator.generate()
infos = scene_bundle.infos
```
The returned `SceneBundle` includes placement metadata (`spawnable_asset_groups`, `spawnable_assets`) so you can re-run or modify clutter/spacing in downstream tooling.

## Testing
Run the module tests from the project root with:
```bash
pytest hazalyser/tests
```
Focus areas include scene generation invariants, helper utilities, and prompt assembly.

## Contribution Guidelines
- Follow the existing code style (PEP 8 with type annotations) and keep comments for non-obvious logic only.
- Place new framework prompts under `frameworks/` and document them.
- Update or add regression tests in `tests/` whenever generator behaviour or controller commands change.

## Support & Further Reading
Consult the design documents in `docs/` and the root project guides for extended workflows, evaluation metrics, and integration guidance.