# hazalyser User Manual

## 1. Introduction
hazalyser is a hazard-analysis toolkit layered on the LEGENT simulator. It generates single-room environments, captures scene imagery, and orchestrates large-language-model (LLM) assessments to classify hazards and record risk narratives. This manual explains how to install, configure, and operate the module.

## 2. Prerequisites
- **Miniconda**: Install Miniconda and create a dedicated conda environment (see [LEGENT installation](https://docs.legent.ai/documentation/getting_started/installation/https://link-url-here.org)).
- **LLM Access**: Optional but required for ESHA analysis. Obtain API keys for your preferred model and set environment variables see Section 4.3.

## 3. Concepts
- **SceneConfig**: Configuration object that captures room specification, subject assets, forced items, and ESHA framework information.
- **SceneGenerator**: Hazard-aware wrapper around LEGENT's generation primitives tailored for single-room ESHA scenarios.
- **SceneBundle**: Return object that packages the generated scene (`infos`) along with placement metadata used for clutter/spacing mutations.
- **Controller**: Chat-style runtime interface that lets you build, adjust, and analyse scenes interactively.
- **Prompter**: Component that parses scenes, builds ESHA prompts, and invokes LLMs.

## 4. Installation & Configuration
### 4.1 Install dependencies
To set up the base simulation environment:

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamjacob97/LEGENT.git
   cd LEGENT
   ```
2. Complete the setup by following the steps provided in the LEGENT documentation: [LEGENT installation](https://docs.legent.ai/documentation/getting_started/installation/https://link-url-here.org)

The documentation includes all required dependencies, environment setup, and platform-specific instructions.

Note: It is recommended to install LEGENT inside a dedicated Conda environment to ensure isolation and reproducibility.
### 4.2 Directory layout
```
hazalyser/
  controller.py
  generator.py
  helpers.py
  house.py
  objects.py
  prompter.py
  smallObjects.py
  obsAssets/
  frameworks/
  llm_analysis/
  tests/
  docs/
```
`frameworks/` holds ESHA prompt templates; `llm_analysis/` stores model responses; `obsAssets/` contains subject/agent meshes.

### 4.3 LLM credentials
Populate environment variables with JSON payloads. Example (`.env` or shell):
```
OPENAI_DEFAULT='{"api_key":"sk-...","base_url":null,"model_name":"gpt-4o","vision_support":true}'
```
During runtime, set `SceneConfig.llm_key="OPENAI_DEFAULT"` so hazalyser can resolve the credentials.

## 5. Quick Start
### 5.1 Programmatic generation
```python
from hazalyser.generator import SceneGenerator, SceneConfig

config = SceneConfig(framework="esha", subject="oldman_earnest.glb" )
generator = SceneGenerator(scene_config=config)
scene_bundle = generator.generate()
infos = scene_bundle.infos
```
`infos` contains LEGENT-compatible scene data. `scene_bundle` also exposes `spawnable_asset_groups` and `spawnable_assets` for downstream edits.

### 5.2 Interactive controller
```python
from hazalyser.controller import Controller
from hazalyser.generator import SceneConfig

controller = Controller(SceneConfig(framework="esha", llm_key="OPENAI_DEFAULT"))
controller.start()
```
In the LEGENT interface, use commands:
- `#NEW` – regenerate scene
- `#CLUTTER+` / `#CLUTTER-` – adjust small-object density
- `#SPACING+` / `#SPACING-` – adjust receptacle spacing
- `#LOCK` – fix current scene for inspection and ESHA
- `#FWANALYSE` – run ESHA prompt using configured LLM key
- `#RESET` / `#UNLOCK` – revert to a previous scene state

### 5.3 Batch ESHA analysis
For scripted pipelines, iterate over generated bundles:
```python
from hazalyser.prompter import Prompter

scene_info = scene_bundle.infos
prompter = Prompter(config, api_key="sk-...", model_name="gpt-4o")
relations = []  # precomputed spatial relations if available
prompter.analyse("hazalyser/llm_analysis/run_001", scene_info, relations)
```
Use `helpers.get_spatial_relations()` if you need live spatial data from the simulator before analysis.

## 6. Advanced Workflows
### 6.1 Forcing items and subjects
Update `SceneConfig.items` with a dictionary (e.g., `{"table": 1, "chair": 2}`) to guarantee those assets. Use `SceneConfig.subject` to inject a subject mesh from `obsAssets/subject`.

### 6.2 Custom ESHA templates
Add a JSON file to `frameworks/` with keys `system_preamble`, `guide`, and `user_instructions`. Reference it via `SceneConfig.framework`. Example structure:
```json
{
  "system_preamble": "You are an ESHA specialist...",
  "guide": "Classify hazards by severity...",
  "user_instructions": "Analyse the provided room layout..."
}
```

### 6.3 Clutter and spacing adjustment
When a scene is locked, `#CLUTTER+/−` and `#SPACING+/−` mutate placements using `SceneBundle` metadata. Changes are reversible until committed with `#COMMIT`.

## 7. Documentation & Diagrams
Refer to `docs/` for detailed design artefacts:
- `scene_generation_high_level_design.md` – narrative pipeline walkthrough
- `scene_gen.pdf` – diagram for the hazard generator
- `uml_hazalyser.pdf` – class relationships
- `User Evaluation Questionnaire.md` – ESHA Evaluation Questionnaire



