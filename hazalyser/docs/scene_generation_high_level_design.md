# Scene Generation Pipeline - High-Level Design

## Purpose
This document explains how `hazalyser.generator.SceneGenerator` builds single-room LEGENT scenes for hazard analysis. It focuses on the procedural generation pipeline inside `hazalyser/generator.py` and omits controller/LLM concerns.

## Key Inputs
- **SceneConfig**: runtime configuration (room spec, agent/subject assets, user-specified items, framework metadata).
- **ObjectDB**: loaded asset catalog (prefabs, placement annotations, asset groups, receptacles, priority types).
- **HazardRoomSpec**: structural template for the target room (type, optional dimension generator).
- **LEGENT primitives**: `HouseGenerator`, `Room`, `Asset`, and `RectPlacer` provide spatial reasoning helpers.

## Primary Outputs
- **SceneBundle**: encapsulates the generated `infos` dict, placement metadata (`spawnable_asset_groups`, `spawnable_assets`), and bookkeeping for clutter/spacing adjustments.
- **infos payload**: the concrete scene description (instances, player/agent placements, room polygons, center point) consumed downstream by controllers and analysis tooling.

## Pipeline Overview
`SceneGenerator.generate()` orchestrates the pipeline in nine ordered phases. Each phase updates shared state (`self.rooms`, `self.placer`) and accumulates final scene instances.

1. **House Structure Synthesis**
   - Calls `generate_structure()` then `generate_house_structure()` to build `HazardHouse` (interior grid, boundary, room polygons).
   - Establishes working bounds (`min_x`, `min_z`, `max_x`, `max_z`) and initialises `RectPlacer` for collision-aware placement across the whole room footprint.

2. **Shell Construction**
   - `add_floors_and_walls()` instantiates floor and wall prefabs for every occupied grid cell, respecting unit scaling and optional wall removal.
   - Returns both the structural instances and a dense floor occupancy array (`floors`) later reused for agent sampling.

3. **Room Registration**
   - Converts `HazardHouse.xz_poly_map` into shapely floor polygons via `get_floor_polygons()` (LEGENT helper) and registers a single `Room` object in `self.rooms`.
   - `Room` stores candidate rectangles, anchors, and asset caches leveraged by subsequent placement routines.

4. **Human and Subject Placement**
   - `add_player_agent_subject()` samples free floor tiles and uses `RectPlacer` to ensure the player avatar, playmate agent, and optional subject mesh occupy non-overlapping positions.
   - Mesh size lookups (`get_mesh_size`) and random yaw allow variation while keeping entities within bounds.

5. **User-Constrained Furniture**
   - `split_items_into_receptacles_and_objects()` partitions requested items into receptacle classes (large floor assets) and small-object prefabs.
   - `place_specified_objects()` consumes receptacle requests: for each type it samples candidate rectangles from the room, tests prefab fit (`prefab_fit_rectangle`), and books space in `RectPlacer` to avoid collisions. Successful placements are tracked as `specified_object_instances` and mark their semantic types in `specified_object_types`.

6. **Background Asset Seeding**
   - `add_other_objects()` prepares the arena for procedural furnishing:
     - Derives `spawnable_asset_groups` by filtering LEGENT asset group tables for the current room type (and respecting priority heuristics).
     - Fetches per-room `spawnable_assets` via `ObjectDB.FLOOR_ASSET_DICT` (floor-eligible prefabs with size and placement metadata).
     - Iteratively samples rectangles and anchors, then calls `sample_and_add_floor_asset()` or `room.sample_place_asset_in_rectangle()` to enqueue primary furniture into `room.assets` while pruning already-used asset types from the pools.

7. **Furniture Materialisation**
   - `place_other_objects()` walks the newly populated `room.assets` list and converts assets (standalone prefabs and group compositions) into concrete scene instances.
   - Each placement is validated with `RectPlacer` to preserve collision constraints and skips prefabs that clash with user-requested types (`specified_object_types`).

8. **Small Object Dressing**
   - Computes `max_object_types_per_room` proportional to clutter limits and calls `add_small_objects()` to scatter lightweight interactables.
   - This helper populates tabletops and receptacles while respecting counts derived from user input and generator heuristics.

9. **Scene Assembly**
   - Concatenates structural, subject, specified, procedural, and small-object instances into a single list.
   - Derives camera centring metadata and room polygon descriptors, then packages everything into `infos` and returns a populated `SceneBundle`.

## Supporting Mechanics
- **Collision Management**: `RectPlacer` coordinates placement across phases (agents, specified furniture, procedural assets). Each successful placement reserves spatial rectangles keyed by prefab or group names.
- **Room Geometry Sampling**: `Room.sample_next_rectangle()` and related anchor helpers expose pre-tiled rectangles to encourage valid layouts relative to walls and room centre lines.
- **Asset Prioritisation**: Priority lists (`ObjectDB.PRIORITY_ASSET_TYPES`) ensure essential furniture types appear before optional clutter. Generated spawn pools dynamically remove non-duplicable types.
- **Randomisation Hooks**: Uniform sampling drives room dimensions (if unspecified), agent positions, furniture selection, and rotations, producing diverse yet plausible scenes.
- **Extensibility**: New placement heuristics fit naturally by extending steps 5-8 (for example, alternative collision policies or thematic asset filters) because the pipeline isolates configuration parsing, asset pooling, and instantiation.

## Data Flow Summary
```
SceneConfig  ->
ObjectDB     ->  SceneGenerator.generate()
                |
                |-> generate_structure() -> HazardHouse geometry
                |-> add_floors_and_walls() -> structural instances + floors grid
                |-> get_rooms() -> Room cache + RectPlacer
                |-> add_player_agent_subject() -> agent instances
                |-> split/place specified objects() -> user furniture
                |-> add_other_objects()/place_other_objects() -> procedural furniture
                |-> add_small_objects() -> accessories
                |-> assemble infos -> SceneBundle
```

## Comparison with Hazalyser SceneGenerator
- **Room Scope**: Core LEGENT supports multi-room houses with door connectivity; `hazalyser.SceneGenerator` targets a single user-specified room without door carving and pads the floorplan manually.
- **Outputs**: Core returns a raw `infos` dict (plus debug file). Hazalyser wraps the payload in `SceneBundle`, exposing clutter/spacing counters and spawnable pools for downstream interactive editing.
- **Configuration Surface**: Hazalyser introduces `SceneConfig` to capture analysis metadata (framework, subject, temperature) and routes subject placement through `add_player_agent_subject`. Core LEGENT operates on `RoomSpec` and external object count overrides without awareness of hazard-analysis parameters.
- **Furniture Pipeline**: Hazalyser pre-splits requested items into receptacles vs small objects and stages them before bulk furnishing, while the core generator expects separate dicts and processes receptacles inline with the per-room loop. Hazalyser caches `spawnable_asset_groups` once per generation; core recomputes the filtered DataFrame per room iteration.
- **Collision Management Enhancements**: Hazalyser reuses `RectPlacer` but applies it to analysis-driven controls (clutter/spacing adjustments) and preserves `SceneBundle` references for future mutations. Core LEGENT focuses on one-shot generation and immediately serialises the result.


## Considerations for Future Work
- **Determinism**: Introduce seeded RNG control to reproduce scenes for testing.
- **Validation**: Add geometry sanity checks (for example, ensure subject placement meets visibility constraints).
- **Multi-room Scaling**: Current pipeline assumes a single room; additional rooms would require `Room` registration, multi-room asset partitioning, and navigation heuristics.

