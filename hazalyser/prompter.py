from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    # Soft dependency; only used if provided
    import openai  # type: ignore  # noqa: F401
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

from legent.scene_generation.objects import get_default_object_db, ObjectDB


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class HazardEntry:
    """Single ESHA worksheet line item."""

    item: str
    category: str  # Environmental Feature | Object | Agent
    location: str
    interaction: str
    failure_keyword: str
    consequence: str
    inherent_measures: str
    safeguards: str
    instructions: str
    severity: Optional[str] = None  # e.g., Low | Medium | High | Critical
    likelihood: Optional[str] = None  # e.g., Rare | Unlikely | Possible | Likely | Frequent
    risk_rating: Optional[str] = None  # e.g., Low | Medium | High | Extreme


@dataclass
class Result:
    """Container for ESHA outputs."""

    summary: Dict[str, Any]
    hazards: List[HazardEntry]
    report_markdown: str
    raw_model_output: Optional[str] = None


# ------------------------------- LLM adapter ------------------------------- #


class _OpenAIAdapter:
    """Minimal adapter over OpenAI chat completions with optional images.

    Uses the messages schema compatible with vision models. Construct only if
    an API key is available and the `openai` package can be imported.
    """

    def __init__(self, api_key: Optional[str], base_url: Optional[str]) -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI client not available.")
        # Lazy import inside to tolerate environments without openai installed
        import openai as _openai  # type: ignore

        if api_key:
            self.client = _openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            # Fallback to default env var handling if available
            self.client = _openai.OpenAI()

    def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        content = response.choices[0].message.content or ""
        return content


# ------------------------------ ESHA Analyzer ------------------------------ #


class Prompter:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        vision_model_name: Optional[str] = None,
    ) -> None:
        self.object_db: ObjectDB = get_default_object_db()
        self.model_name = model_name
        # Use a dedicated vision model if provided; otherwise reuse model_name
        self.vision_model_name = vision_model_name or model_name

        self._llm: Optional[_OpenAIAdapter] = None
        if _OPENAI_AVAILABLE:
            try:
                self._llm = _OpenAIAdapter(api_key=api_key, base_url=base_url)
            except Exception:
                # No-op; fallback path will be used
                self._llm = None

    # ------------------------------ Public API ------------------------------ #

    def analyze(
        self,
        scene_dict: Dict[str, Any],
        images: Optional[Sequence[str]] = None,
        extra_context: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Result:

        parsed = self._parse_scene(scene_dict)
        metrics = self._compute_scene_metrics(parsed)
        prompt_blocks = self._build_prompt(parsed, metrics, extra_context)

        # Try LLM path first, with rule-based fallback
        if self._llm is not None:
            try:
                messages = self._build_messages(prompt_blocks, images)
                output = self._llm.chat(
                    model=(self.vision_model_name if images else self.model_name),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                hazards = self._extract_hazards_from_text(output)
                report_md = self._compose_report_markdown(parsed, metrics, hazards, llm_markdown=output)
                return Result(
                    summary={"parsed": parsed, "metrics": metrics},
                    hazards=hazards,
                    report_markdown=report_md,
                    raw_model_output=output,
                )
            except Exception as e:
                # Fall through to rule-based if LLM fails
                fallback = self._rule_based_hazards(parsed, metrics, error=str(e))
                report_md = self._compose_report_markdown(parsed, metrics, fallback, llm_markdown=None)
                return Result(
                    summary={"parsed": parsed, "metrics": metrics, "note": "LLM unavailable; rule-based fallback used."},
                    hazards=fallback,
                    report_markdown=report_md,
                    raw_model_output=None,
                )
        else:
            hazards = self._rule_based_hazards(parsed, metrics, error=None)
            report_md = self._compose_report_markdown(parsed, metrics, hazards, llm_markdown=None)
            return Result(
                summary={"parsed": parsed, "metrics": metrics, "note": "LLM not configured; rule-based fallback used."},
                hazards=hazards,
                report_markdown=report_md,
                raw_model_output=None,
            )

    # ----------------------------- Scene Parsing ---------------------------- #

    def _parse_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected keys in scene dict (as produced by generator):
          - instances: List[Dict]
          - room_polygon: List[Dict] with room_id, room_type, position, size, polygon
          - player: Dict
          - agent: Dict
          - center: [x,y,z]
        """
        instances: List[Dict[str, Any]] = scene.get("instances", [])
        rooms: List[Dict[str, Any]] = scene.get("room_polygon", [])
        player = scene.get("player", {})
        agent = scene.get("agent", {})

        # Index instances by room_id
        room_id_to_instances: Dict[int, List[Dict[str, Any]]] = {}
        unassigned_instances: List[Dict[str, Any]] = []
        for inst in instances:
            room_id = inst.get("room_id")
            if isinstance(room_id, int):
                room_id_to_instances.setdefault(room_id, []).append(inst)
            else:
                # Handle instances without valid room_id (e.g., global items like player/agent markers)
                unassigned_instances.append(inst)

        # Basic counts
        floors = [i for i in instances if self._is_floor(i)]
        walls = [i for i in instances if self._is_wall(i)]
        doors = [i for i in instances if self._is_door(i)]

        # Map prefab to semantic type if known
        enriched_instances: List[Dict[str, Any]] = []
        for i in instances:
            prefab = i.get("prefab")
            if prefab:
                type_name = self.object_db.OBJECT_TO_TYPE.get(prefab, "")
            else:
                type_name = ""
            enriched = dict(i)
            enriched["object_type"] = type_name
            enriched_instances.append(enriched)

        parsed = {
            "rooms": rooms,
            "instances": enriched_instances,
            "room_id_to_instances": room_id_to_instances,
            "unassigned_instances": unassigned_instances,
            "floors": floors,
            "walls": walls,
            "doors": doors,
            "player": player,
            "agent": agent,
        }
        return parsed

    # ------------------------------ Heuristics ------------------------------ #

    def _is_floor(self, inst: Dict[str, Any]) -> bool:
        prefab = (inst.get("prefab") or "").lower()
        return "floor" in prefab

    def _is_wall(self, inst: Dict[str, Any]) -> bool:
        prefab = (inst.get("prefab") or "").lower()
        return "wall" in prefab

    def _is_door(self, inst: Dict[str, Any]) -> bool:
        prefab = (inst.get("prefab") or "").lower()
        return "door" in prefab

    # ------------------------------ Metrics -------------------------------- #

    def _compute_scene_metrics(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        rooms = parsed["rooms"]
        room_metrics: List[Dict[str, Any]] = []

        # Handle edge case of no rooms
        if not rooms:
            # Create a default "global" room from all instances
            total_instances = parsed["instances"]
            global_room_metrics = {
                "room_id": "global",
                "room_type": "Unknown",
                "area_m2": 0.0,  # Unknown area
                "num_instances": len(total_instances),
                "num_movable": sum(1 for i in total_instances if self._is_potentially_movable(i)),
                "num_receptacles": sum(1 for i in total_instances if i.get("type") == "receptacle"),
                "num_doors": len(parsed["doors"]),
                "num_walls": len(parsed["walls"]),
                "clutter_density_per_m2": 0.0,  # Cannot compute without area
                "movable_density_per_m2": 0.0,
            }
            room_metrics.append(global_room_metrics)
        else:
            for r in rooms:
                room_id = r.get("room_id")
                room_type = r.get("room_type")
                size = r.get("size", [0, 0, 0])
                x_size, _, z_size = size if len(size) == 3 else (0, 0, 0)
                area = max(0.0, float(x_size) * float(z_size))
                instances = parsed["room_id_to_instances"].get(room_id, [])
                movable_count = sum(1 for i in instances if self._is_potentially_movable(i))
                receptacle_count = sum(1 for i in instances if i.get("type") == "receptacle")
                door_count = sum(1 for i in instances if self._is_door(i))
                wall_count = sum(1 for i in instances if self._is_wall(i))
                clutter_density = self._safe_div(len(instances), area)
                movable_density = self._safe_div(movable_count, area)

                room_metrics.append(
                    {
                        "room_id": room_id,
                        "room_type": room_type,
                        "area_m2": round(area, 3),
                        "num_instances": len(instances),
                        "num_movable": movable_count,
                        "num_receptacles": receptacle_count,
                        "num_doors": door_count,
                        "num_walls": wall_count,
                        "clutter_density_per_m2": round(clutter_density, 3),
                        "movable_density_per_m2": round(movable_density, 3),
                    }
                )

        totals = {
            "num_rooms": len(rooms),
            "num_instances": len(parsed["instances"]),
            "num_doors": len(parsed["doors"]),
            "num_walls": len(parsed["walls"]),
            "num_floors": len(parsed["floors"]),
        }

        return {"rooms": room_metrics, "totals": totals}

    def _is_potentially_movable(self, inst: Dict[str, Any]) -> bool:
        # Heuristic: if not floor/wall/door/receptacle, consider small object movable
        if self._is_floor(inst) or self._is_wall(inst) or self._is_door(inst):
            return False
        if inst.get("type") == "receptacle":
            return False
        return True

    def _safe_div(self, a: float, b: float) -> float:
        return a / b if b not in (0, 0.0, None) else 0.0

    # ---------------------------- Prompt Building --------------------------- #

    def _build_prompt(
        self,
        parsed: Dict[str, Any],
        metrics: Dict[str, Any],
        extra_context: Optional[str],
    ) -> Dict[str, Any]:
        # Compose a compact structured summary to feed the LLM
        summary = {
            "rooms": metrics["rooms"],
            "totals": metrics["totals"],
            "agents": {
                "player": parsed.get("player"),
                "agent": parsed.get("agent"),
            },
            "sample_instances": self._sample_instances(parsed["instances"], k=40),
        }

        system_preamble = (
            "You are an expert safety analyst performing ESHA (Environmental Survey "
            "Hazard Analysis) for autonomous mobile robots. ESHA focuses on "
            "identifying hazards from both mission and non-mission interactions "
            "with environmental features, objects, and agents. Output a thorough "
            "but concise analysis referencing the scene summary."
        )

        guide = (
            "ESHA Taxonomy and Expectations:\n"
            "- Environmental Features: fixed background, terrain, surfaces, ambient.\n"
            "- Objects: embedded items (0D/1D/2D/3D), stationary immovable, stationary movable, or moving non-purposefully.\n"
            "- Agents: purposeful motion (humans, robots, animals).\n"
            "Report: Provide a Generic ESHA Worksheet (table) with columns: Item, Category, Location, Interaction, Failure Keyword, Consequence, Inherent Measures, Safeguards, Instructions, Severity, Likelihood, Risk. Then provide narrative rationale and safety requirements."
        )

        scene_json = json.dumps(summary, ensure_ascii=False, indent=2)

        user_instructions = (
            "Analyze the provided scene summary (and images if present). "
            "Identify both mission and non-mission interactions and enumerate all "
            "reasonably foreseeable hazards. Consider collision, entrapment, "
            "visibility/occlusion, tipping/overturning, access/egress, doorway bottlenecks, "
            "stairs/elevation changes (if any), clutter density relative to area, and "
            "interaction with movable vs. immovable objects. Provide mitigation measures in all three categories."
        )

        if extra_context:
            user_instructions += f"\nExtra context: {extra_context}"

        return {
            "system": system_preamble,
            "guide": guide,
            "scene": scene_json,
            "user": user_instructions,
        }

    def _sample_instances(self, instances: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        # Down-sample to keep prompt compact; keep representative spread
        if len(instances) <= k:
            return instances
        # Simple strided sample
        stride = max(1, len(instances) // k)
        return instances[::stride][:k]

    def _build_messages(self, prompt: Dict[str, Any], images: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": prompt["guide"]}]},
            {"role": "user", "content": [{"type": "text", "text": "Scene Summary:"}]},
            {"role": "user", "content": [{"type": "text", "text": prompt["scene"]}]},
            {"role": "user", "content": [{"type": "text", "text": prompt["user"]}]},
        ]

        if images:
            # Attach up to 6 images to keep requests small
            for image_path in list(images)[:6]:
                try:
                    encoded = self._encode_image_file(image_path)
                    # Detect image format from file extension
                    ext = os.path.splitext(image_path)[1].lower()
                    mime_type = "image/png"  # default
                    if ext in ['.jpg', '.jpeg']:
                        mime_type = "image/jpeg"
                    elif ext in ['.webp']:
                        mime_type = "image/webp"
                    elif ext in ['.gif']:
                        mime_type = "image/gif"
                    
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

    # --------------------------- Hazard Extraction -------------------------- #

    def _extract_hazards_from_text(self, text: str) -> List[HazardEntry]:
        """Best-effort parser: expects a table-like section in markdown.

        If parsing fails, return an empty list and rely on the report markdown.
        """
        hazards: List[HazardEntry] = []
        lines = [line.strip() for line in text.splitlines()]
        # Find a simple pipe-table block and parse rows
        table_lines: List[str] = []
        in_table = False
        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                table_lines.append(line)
                in_table = True
            elif in_table and line == "":
                break
        if not table_lines:
            return hazards

        # Skip header and divider if present
        data_rows = []
        for line in table_lines:
            stripped = line.replace("|", "").replace("-", "").replace(" ", "")
            if not stripped:  # Empty line after removing table chars
                continue
            data_rows.append(line)
        
        # Remove header row and divider row if they exist
        if len(data_rows) >= 2:
            # Check if second row is a divider (contains only |, -, and spaces)
            second_row = data_rows[1].replace(" ", "")
            if set(second_row) <= {"|", "-"}:
                data_rows = data_rows[2:]  # Skip header and divider
            else:
                data_rows = data_rows[1:]  # Skip just header
        elif len(data_rows) >= 1:
            data_rows = data_rows[1:]  # Skip header
        for row in data_rows:
            cols = [c.strip() for c in row.strip("|").split("|")]
            if len(cols) < 9:  # Minimum required columns
                continue
            try:
                hazards.append(
                    HazardEntry(
                        item=cols[0] if len(cols) > 0 else "",
                        category=cols[1] if len(cols) > 1 else "",
                        location=cols[2] if len(cols) > 2 else "",
                        interaction=cols[3] if len(cols) > 3 else "",
                        failure_keyword=cols[4] if len(cols) > 4 else "",
                        consequence=cols[5] if len(cols) > 5 else "",
                        inherent_measures=cols[6] if len(cols) > 6 else "",
                        safeguards=cols[7] if len(cols) > 7 else "",
                        instructions=cols[8] if len(cols) > 8 else "",
                        severity=cols[9] if len(cols) > 9 else None,
                        likelihood=cols[10] if len(cols) > 10 else None,
                        risk_rating=cols[11] if len(cols) > 11 else None,
                    )
                )
            except Exception:
                continue
        return hazards

    # --------------------------- Rule-based Fallback ------------------------ #

    def _rule_based_hazards(
        self,
        parsed: Dict[str, Any],
        metrics: Dict[str, Any],
        error: Optional[str],
    ) -> List[HazardEntry]:
        hazards: List[HazardEntry] = []

        # Doorway congestion
        total_doors = metrics["totals"].get("num_doors", 0)
        if total_doors <= 1 and metrics["totals"].get("num_rooms", 1) >= 1:
            hazards.append(
                HazardEntry(
                    item="Doorway",
                    category="Environmental Feature",
                    location="Global",
                    interaction="Access/Egress",
                    failure_keyword="Bottleneck",
                    consequence="Delayed evacuation or path planning failure",
                    inherent_measures="Prefer wider/dual doors when possible",
                    safeguards="Dynamic re-routing; doorway occupancy monitoring",
                    instructions="Avoid blocking doorways; maintain clear egress",
                    severity="Medium",
                    likelihood="Possible",
                    risk_rating="Medium",
                )
            )

        # High clutter density
        for r in metrics["rooms"]:
            if r["clutter_density_per_m2"] > 1.0:
                hazards.append(
                    HazardEntry(
                        item=f"Room {r['room_id']} ({r['room_type']})",
                        category="Object",
                        location=f"Room {r['room_id']}",
                        interaction="Navigation",
                        failure_keyword="Collision/Entrapment",
                        consequence="Increased collision risk and path blockage",
                        inherent_measures="Lower mass/soft bumpers on platform",
                        safeguards="Speed limiting; obstacle inflation in planner",
                        instructions="Reduce clutter; maintain housekeeping",
                        severity="High",
                        likelihood="Likely",
                        risk_rating="High",
                    )
                )

        # Movable object interaction
        for r in metrics["rooms"]:
            if r["num_movable"] >= 5:
                hazards.append(
                    HazardEntry(
                        item=f"Movable objects in Room {r['room_id']}",
                        category="Object",
                        location=f"Room {r['room_id']}",
                        interaction="Pushing/Displacement",
                        failure_keyword="Secondary hazards",
                        consequence="Movables shift into path or create trip hazards",
                        inherent_measures="Limit pushing force; rounded edges",
                        safeguards="Contact detection; stop-on-push; replan",
                        instructions="Secure or reduce movable items",
                        severity="Medium",
                        likelihood="Possible",
                        risk_rating="Medium",
                    )
                )

        # Human-agent proximity (if both present)
        if parsed.get("player") and parsed.get("agent"):
            hazards.append(
                HazardEntry(
                    item="Human-Agent Interaction",
                    category="Agent",
                    location="Global",
                    interaction="Co-presence",
                    failure_keyword="Near-miss/Collision",
                    consequence="Personal injury or unsafe interaction",
                    inherent_measures="Low max force; compliant surfaces",
                    safeguards="Proximity sensing; speed/force limiting",
                    instructions="Maintain separation; announce movements",
                    severity="High",
                    likelihood="Possible",
                    risk_rating="High",
                )
            )

        # If LLM error, record as note
        if error:
            hazards.append(
                HazardEntry(
                    item="Analysis Engine",
                    category="Environmental Feature",
                    location="N/A",
                    interaction="N/A",
                    failure_keyword="LLM unavailable",
                    consequence="Fallback to rule-based analysis",
                    inherent_measures="Provide offline heuristics",
                    safeguards="Retry with cached prompts",
                    instructions="Configure API key and model",
                    severity="Low",
                    likelihood="Possible",
                    risk_rating="Low",
                )
            )

        return hazards

    # ------------------------------ Reporting ------------------------------- #

    def _compose_report_markdown(
        self,
        parsed: Dict[str, Any],
        metrics: Dict[str, Any],
        hazards: List[HazardEntry],
        llm_markdown: Optional[str],
    ) -> str:
        header = "**Environmental Survey Hazard Analysis (ESHA) Report**\n\n"

        scene_summary = (
            "### Scene summary\n"
            f"- Rooms: {metrics['totals']['num_rooms']}\n"
            f"- Instances: {metrics['totals']['num_instances']} (doors: {metrics['totals']['num_doors']}, walls: {metrics['totals']['num_walls']})\n"
        )

        per_room = "\n".join(
            [
                (
                    f"- Room {r['room_id']} ({r['room_type']}), area {r['area_m2']} m^2: "
                    f"instances={r['num_instances']}, movable={r['num_movable']}, "
                    f"clutter_density={r['clutter_density_per_m2']}/m^2"
                )
                for r in metrics["rooms"]
            ]
        )
        room_block = f"### Rooms\n{per_room}\n\n"

        # Worksheet table
        table_header = (
            "### Generic ESHA Worksheet\n"
            "| Item | Category | Location | Interaction | Failure Keyword | Consequence | Inherent Measures | Safeguards | Instructions | Severity | Likelihood | Risk |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        table_rows = "\n".join(
            [
                (
                    f"| {h.item} | {h.category} | {h.location} | {h.interaction} | {h.failure_keyword} | {h.consequence} | "
                    f"{h.inherent_measures} | {h.safeguards} | {h.instructions} | {h.severity or ''} | {h.likelihood or ''} | {h.risk_rating or ''} |"
                )
                for h in hazards
            ]
        )

        narrative = ""
        if llm_markdown:
            narrative = f"\n### LLM Narrative Analysis\n{llm_markdown}\n"

        return header + scene_summary + room_block + table_header + table_rows + narrative


# ------------------------------ Convenience API ---------------------------- #


def run_llm_analysis(
    scene_dict: Dict[str, Any],
    images: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Result:
    analyzer = Prompter(api_key=api_key, base_url=base_url)
    return analyzer.analyze(scene_dict=scene_dict, images=images)


