ESHA Module for LEGENT

Purpose: Generate and control single-room scenes in LEGENT using ESHA (Environmental Survey Hazard Analysis) workflows. The module wraps LEGENT's native procedural scene generation and assets, and exposes study-oriented controls.

Quick start

```python
from esha import ESHAController

# Start environment and controller
ctrl = ESHAController()

# 1) Generate candidate layouts for a Bedroom with only specified items
items = {"orange": 3, "table": 1, "pumpkin": 2}
cands = ctrl.generate_candidates(
    room_type="Bedroom",
    items=items,
    include_other_items=False,   # only keep specified items plus structure
    num_candidates=3,
)

# 2) Preview each candidate visually
for sc in cands:
    ctrl.preview_scene(sc, seconds=3)

# 3) Lock one scene (e.g., the first)
ctrl.lock_scene(cands[0])

# 4) Post-selection controls
ctrl.level_of_clutter(0.8)            # denser small objects
ctrl.spacing(0.4)                     # increase minimum spacing between small objects
ctrl.arrange_by("proximity", "table") # cluster around a table
ctrl.remove_item("orange", 1)         # remove one orange
ctrl.add_item("pumpkin", 1)           # add one pumpkin

# Cleanup when done
ctrl.close()
```

Key API

- generate_candidates(room_type, items=None, include_other_items=True, num_candidates=3, dims=None):
  - Single-room procedural generation with optional item dict.
  - items keys must be native LEGENT types (e.g., "orange") or prefab names.
  - include_other_items=False keeps only specified items plus floors/walls/doors.

- preview_scene(scene, seconds=3.0):
  - Loads and steps the environment briefly for visual inspection.

- lock_scene(scene):
  - Locks a chosen scene and makes it the target of subsequent controls.

Controls after locking

- level_of_clutter(value: float): 0.0 (sparse) to 1.0 (dense), adjusts small objects.
- spacing(value: float): enforces minimum spacing between small objects.
- arrange_by(parameter: str, reference: Optional[str] = None):
  - "proximity": cluster smalls near a reference type/prefab.
  - "category": cluster by object types in quadrants.
- remove_item(item_name: str, count: int): remove matched prefab/type instances.
- add_item(item_name: str, count: int): add smalls by type/prefab, fallback to floor assets when applicable.

Notes

- The module reuses LEGENT scene generation (HouseGenerator) and assets. It respects placement rules via RectPlacer to avoid collisions.
- Item names must match LEGENT's native types (lowercase) in OBJECT_DICT (e.g., "orange", "pumpkin", "table") or be exact prefab names from PREFABS.
- Some arrangements are heuristic and prioritize speed and visual control; they remain compatible with LEGENT physics/rendering.


ESHA Analyzer
-------------

The ESHA Analyzer performs Environmental Survey Hazard Analysis on a scene dict returned by LEGENT. It supports text-only analysis and optional images using a vision-capable LLM. It does not modify LEGENT core.

Minimal usage:

```python
from esha import ESHAController, ESHAAnalyzer, run_esha_analysis

# 1) Generate a scene
ctrl = ESHAController()
scene = ctrl.generate_candidate()

# 2a) Analyze (rule-based fallback if no OpenAI configured)
analyzer = ESHAAnalyzer()  # optionally pass api_key, base_url
result = analyzer.analyze(scene_dict=scene, images=None, extra_context="Warehouse robot")
print(result.report_markdown)

# 2b) Or use one-shot helper
result2 = run_esha_analysis(scene_dict=scene)
```

Outputs:

- result.summary: scene/metrics summary used in the analysis
- result.hazards: structured list of hazards (Generic ESHA Worksheet rows)
- result.report_markdown: complete ESHA report (worksheet table + narrative)

OpenAI configuration (optional):

- If OpenAI is installed and API credentials are available (env or passed), the analyzer uses an LLM (e.g., `gpt-4o-mini`) and a vision model when images are included.
- Without OpenAI, a robust rule-based fallback is used to produce a baseline ESHA.

