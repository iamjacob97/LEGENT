import base64
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

from hazalyser.helpers import is_small_object_instance, is_subject, load_framework_prompt
from legent.scene_generation.objects import get_default_object_db, ObjectDB

class _OpenAIAdapter:
    """Minimal adapter over OpenAI chat completions with optional images.

    Uses the messages schema compatible with vision models. Construct only if
    an API key is available and the `openai` package can be imported.
    """
    def __init__(self, api_key: Optional[str], base_url: Optional[str]) -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI client not available. Please download the openai library in this environment")
        if api_key:
            self.client = _openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            # Fallback to default env var handling if available
            self.client = _openai.OpenAI()

    def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content or ""
        usage = response.usage
        return content, usage

class Prompter:
    def __init__(self, scene_config, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                model_name: str = "", vision_model: bool= False) -> None:
        self.odb: ObjectDB = get_default_object_db()
        self.scene_config = scene_config
        self.model_name = model_name

        self._llm: Optional[_OpenAIAdapter] = None
        if _OPENAI_AVAILABLE:
            try:
                self._llm = _OpenAIAdapter(api_key=api_key, base_url=base_url)
            except Exception as e:
                raise RuntimeError(f"LLM Adapter not properly initialized: {e}")
    # ------------------------------ Public API ------------------------------ #

    def analyze(self, save_path, scene_info: Dict[str, Any], spatial_ralations, images: Optional[Sequence[str]] = None, 
                temperature: float = 0.2, max_output_tokens: int = None):

        parsed = self._parse_scene(scene_info, spatial_ralations)
        metrics = self._compute_scene_metrics(parsed)
        prompt_blocks = self._build_prompt(parsed, metrics)

        messages = self._build_messages(prompt_blocks, images)
        output, usage = self._llm.chat(model=self.model_name, messages=messages,
                                       temperature=temperature, max_tokens=max_output_tokens)
        
        self._save_llm_output(output, usage, save_path, temperature, max_output_tokens)
        
    # ----------------------------- Scene Parsing ---------------------------- #

    def _parse_scene(self, scene_info: Dict[str, Any], spatial_raltions) -> Dict[str, Any]:
        """
        Expected keys in scene dict (as produced by generator):
          - instances: List[Dict]
          - room_polygon: List[Dict] with room_id, room_type, position, size, polygon
          - player: Dict
          - agent: Dict
          - center: [x,y,z]
        """
        instances: List[Dict[str, Any]] = scene_info.get("instances", [])
        room: List[Dict[str, Any]] = scene_info.get("room_polygon", [])
        player = scene_info.get("player", {})
        agent = scene_info.get("agent", {})
        subject = None
        for instance in instances:
            if is_subject(instance, self.scene_config.subject):
                subject = instance
                break
        instances.remove(subject)

        # Basic counts
        floors = [i for i in instances if self._is_floor(i)]
        walls = [i for i in instances if self._is_wall(i)]

        # Map prefab to semantic type if known
        enriched_instances: List[Dict[str, Any]] = []
        for i in instances:
            prefab = i.get("prefab")
            if prefab:
                type_name = self.odb.OBJECT_TO_TYPE.get(prefab, "")
            else:
                type_name = ""
            enriched = dict(i)
            enriched["object_type"] = type_name
            enriched_instances.append(enriched)

        parsed = {
            "room": room,
            "instances": enriched_instances,
            "relations": spatial_raltions,
            "floors": floors,
            "walls": walls,
            "player": player,
            "agent": agent,
            "subject": subject,
        }
        return parsed

    def _is_floor(self, inst: Dict[str, Any]) -> bool:
        prefab = (inst.get("prefab") or "").lower()
        structural_assets = self.odb.MY_OBJECTS
        floor_prefabs = set(structural_assets.get("floor", []))
        return prefab in floor_prefabs

    def _is_wall(self, inst: Dict[str, Any]) -> bool:
        prefab = (inst.get("prefab") or "").lower()
        structural_assets = self.odb.MY_OBJECTS
        wall_prefabs = set(structural_assets.get("wall", []))
        return prefab in wall_prefabs

    # ------------------------------ Metrics -------------------------------- #

    def _compute_scene_metrics(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        room = parsed["room"]
        strucutural_assets = self.odb.MY_OBJECTS
        wall_prefabs = set(strucutural_assets.get("wall", []))
        floor_prefabs = set(strucutural_assets.get("floor", []))
        walls_and_floors = wall_prefabs | floor_prefabs
        
        r = room[0]
        room_id = r.get("room_id")
        room_type = r.get("room_type")
        size = r.get("size", [0, 0, 0])
        x_size, _, z_size = size if len(size) == 3 else (0, 0, 0)
        area = max(0.0, float(x_size) * float(z_size))
        instances = parsed["instances"]
        movable_count = 0
        receptacle_count = 0
        for instance in instances:
            if is_small_object_instance(instance):
                movable_count += 1
            elif instance not in walls_and_floors and not is_subject(instance, self.scene_config.subject):
                receptacle_count += 1
        clutter_density = self._safe_div(len(instances), area)

        return {
            "room_id": room_id,
            "room_type": room_type,
            "area_m2": round(area, 3),
            "num_instances": len(instances),
            "num_walls": len(parsed["walls"]),
            "num_floors": len(parsed["floors"]),
            "num_movable": movable_count,
            "num_receptacles": receptacle_count,
            "clutter_density_per_m2": round(clutter_density, 3),
        }

    def _safe_div(self, a: float, b: float) -> float:
        return a / b if b not in (0, 0.0, None) else 0.0

    # ---------------------------- Prompt Building --------------------------- #

    def _build_prompt(self, parsed: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        scene_config = self.scene_config
        # Compose a compact structured summary to feed the LLM
        summary = {
            "instances": parsed["instances"],
            "relations": parsed["relations"],
            "agents": {
                "player": parsed.get("player"),
                "agent": parsed.get("agent"),
                "subject": parsed.get("subject")
            },
            "metrics": metrics,
        }
        
        prompt_blocks = load_framework_prompt(self.scene_config)
        system_preamble = prompt_blocks["system_preamble"]
        guide = prompt_blocks["guide"]
        scene_json = json.dumps(summary, ensure_ascii=False, indent=2)
        user_instructions = prompt_blocks["user_instructions"]
        if scene_config.agent_info or scene_config.subject_info or scene_config.task:
            extra_context = f"agent_info: {scene_config.agent_info}, subject_info: {scene_config.subject_info}, task_info: {scene_config.task}"
            user_instructions += f"\nExtra context: {extra_context}"

        return {
            "system": system_preamble,
            "guide": guide,
            "scene": scene_json,
            "user": user_instructions,
        }

    def _build_messages(self, prompt: Dict[str, Any], images: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": prompt["guide"]}]},
            {"role": "user", "content": [{"type": "text", "text": "Scene Summary:"}]},
            {"role": "user", "content": [{"type": "text", "text": prompt["scene"]}]},
            {"role": "user", "content": [{"type": "text", "text": prompt["user"]}]},
        ]

        if self.scene_config.vision_model and images:
            # Attach up to 6 images to keep requests small
            for image_path in list(images):
                try:
                    encoded = self._encode_image_file(image_path)
                    mime_type = "image/png"  # default
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Image: {image_path}"},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                            ],
                        }
                    )
                except Exception:
                    # Skip unreadable image
                    continue
        return messages

    def _encode_image_file(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ------------------------------ Reporting ------------------------------- #

    def _save_llm_output(self, output, usage, save_path, temperature, max_output_tokens):
        scene_config = self.scene_config
        timestamp = datetime.now().strftime("%d%m%Y_%H%M")

        analysis_file = os.path.join(save_path, f"{scene_config.framework}_{scene_config.llm_key}_{timestamp}.txt")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write("=== LLM OUTPUT ===\n")
            f.write(output)
            f.write("\n\n=== USAGE DETAILS ===\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Max tokens: {max_output_tokens}\n")
            f.write(f"Prompt tokens: {usage.prompt_tokens}")
            f.write(f"Completion tokens: {usage.completion_tokens}")
            f.write(f"Total tokens: {usage.total_tokens}")
            