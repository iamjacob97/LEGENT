import copy
import re

from hazalyser.smallObjects import add_small_objects
from hazalyser.utils import deepcopy_scene
from hazalyser.generator import SceneConfig, SceneBundle, SceneGenerator
from hazalyser.helpers import get_current_scene_state, update_position_and_rotation, save_perspectives
from legent import Environment, ResetInfo, Action

class Controller:
    """
    Controller orchestrates single-room scene generation and interactive control in LEGENT.

    Design goals:
    - Single-room focus with user-chosen room type
    - Optional item dictionary to force specific items (receptacles and/or small objects)
    - Post-selection controls: clutter, spacing, add/remove items
    - Zero edits to LEGENT internals; reuse its generators and assets
    - Chat-controlled workflow: "lock_scene()", "reset_scene()", "quit()"
    """

    def __init__(self, scene_config = SceneConfig()) -> None:
        self.scene_config: SceneConfig = scene_config
        self._locked_scene_bundle: SceneBundle = None
        self._current_scene_bundle: SceneBundle = None
        self._running: bool = False
        self.marked_scenes = []
        self.command_handlers = {
            "#LOCK": self._lock_scene,
            "#UNLOCK": self._unlock_scene,
            "#RESET": self._reset_scene,
            "#COMMIT": self._commit_scene,
            "#RELOAD": self._reload_scene,
            "#CLUTTER+": self._increase_clutter,
            "#CLUTTER-": self._decrease_clutter,
            "#SPACING+": self._increase_spacing,
            "#SPACING-": self._decrease_spacing,
            "#ANALYSE": self._analyse
        }
    
    def start(self, env_path: str = "auto") -> None:
        """
        Start the Controller.
        """
        self._running = True
        env = Environment(env_path=env_path)
        action = Action()
        marked = False

        try:
            self._new_scene(env)
            while self._running:
                obs = env.step()
                cmd = (obs.text or "").strip()

                if cmd in self.command_handlers:
                    self.command_handlers[cmd](env, action)

                elif cmd == "#MARK":
                    if marked:
                        action.text = "Scene already marked."
                        env.step(action)
                    else:
                        action.text = f"Scene {len(self.marked_scenes)} marked."
                        self.marked_scenes.append(self._current_scene_bundle)
                        marked = True
                        env.step(action)

                elif cmd == "#LIST":
                    if self._locked_scene_bundle:
                        action.text = "Scene locked. Cannot list marked scenes."
                        env.step(action)
                    else:
                        if not self.marked_scenes:
                            action.text = "No scenes marked."
                            env.step(action)
                        else:
                            action.text = f"Entering marked scenes, there are {len(self.marked_scenes)} scenes marked."
                            env.step(action)
                            marked = True
                            self._enter_list_view(env, action)
                
                elif cmd == "#CLEAR":
                    self.marked_scenes.clear()
                    action.text = "Marked scenes cleared."
                    marked = False
                    env.step(action)
                
                elif cmd == "#NEW":
                    if not self._locked_scene_bundle:
                        self._new_scene(env)
                        marked = False
                    else:
                        action.text = "Scene already locked. Use #RESET to reset the locked scene or #UNLOCK to unlock the scene."
                        env.step(action)

                elif cmd == "#QUIT":
                    action.text = "Exiting"
                    env.step(action)
                    self._running = False
        finally:
            env.close()

        # ---------- Internals ----------

    def _new_scene(self, env: Environment) -> None:
        scene_generator = SceneGenerator(scene_config=self.scene_config)
        scene_bundle = scene_generator.generate()
        scene = scene_bundle.infos
        self._current_scene_bundle = scene_bundle
        env.reset(ResetInfo(scene=scene))    

    def _show_scene(self, env: Environment, action: Action, index: int) -> None:
        """Helper to show a specific scene."""
        scene_num = index % len(self.marked_scenes)
        scene_bundle = self.marked_scenes[scene_num]
        self._current_scene_bundle = scene_bundle
        scene = scene_bundle.infos
        env.reset(ResetInfo(scene=scene))
    
        action.text = f"You are now in Scene{scene_num}"
        env.step(action)
    
    def _enter_list_view(self, env: Environment, action: Action) -> None:
        """Enter the marked scenes browsing mode."""
        current_index = 0
        self._show_scene(env, action, current_index)

        while True:
            obs = env.step()
            cmd = (obs.text or "").strip()

            if cmd in self.command_handlers:
                self.command_handlers[cmd](env, action)
            
            elif cmd == "#NEXT":
                if self._locked_scene_bundle:
                    action.text = "Scene locked. Cannot go to next."
                    env.step(action)
                else:
                    current_index += 1
                    self._show_scene(env, action, current_index)

            elif cmd == "#PREV":
                if self._locked_scene_bundle:
                    action.text = "Scene locked. Cannot go to previous."
                    env.step(action)
                else:
                    current_index -= 1
                    self._show_scene(env, action, current_index)
            
            elif cmd.startswith("#SCENE"):
                scene_num = re.search(r"\d+", cmd)
                if scene_num:
                    if self._locked_scene_bundle:
                        action.text = "Scene locked. Cannot go to specific scene."
                        env.step(action)
                    else:
                        scene_num = int(scene_num.group())
                        if 0 <= scene_num < len(self.marked_scenes):
                            current_index = scene_num
                            self._show_scene(env, action, current_index)
                        else:
                            action.text = "Scene number out of bounds."
                            env.step(action)
                else:
                    action.text = "Invalid scene number."
                    env.step(action)

            elif cmd == "#REMOVE":
                if self._locked_scene_bundle:
                    action.text = "Scene locked. Cannot remove."
                    env.step(action)
                else:
                    scene_num = current_index % len(self.marked_scenes)
                    self.marked_scenes.pop(scene_num)
                    action.text = "Scene removed."
                    env.step(action)
                    if self.marked_scenes:
                        self._show_scene(env, action, scene_num)
                    else:
                        action.text = "No scenes marked. Exiting marked scenes."
                        env.step(action)
                        return

            elif cmd == "#BACK":
                action.text = "Exiting marked scenes."
                env.step(action)
                return

            elif cmd == "#QUIT":
                action.text = "Exiting"
                env.step(action)
                self._running = False
                return

    def _reset_scene(self, env: Environment, action: Action) -> None:
        env.reset(ResetInfo(scene=self._current_scene_bundle))
        action.text = "Scene reset."
        env.step(action)
        
    def _lock_scene(self, env: Environment, action: Action) -> None:
        self._locked_scene_bundle = deepcopy_scene(self._current_scene_bundle)
        action.text = "Scene locked."
        env.step(action)

    def _unlock_scene(self, env: Environment, action: Action) -> None:
        self._locked_scene_bundle = None
        action.text = "Scene unlocked."
        env.step(action)

    def _commit_scene(self, env: Environment, action: Action) -> None:
        """Save current scene state of the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot save."
            env.step(action)
        else:
            current_state = get_current_scene_state(env)
            update_position_and_rotation(self._locked_scene_bundle.infos, current_state)

            action.text = "Scene saved."
            env.step(action)
    
    def  _reload_scene(self, env: Environment, action: Action) -> None:
        """Reload the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot reload."
            env.step(action)
        else:
            env.reset(ResetInfo(scene=self._locked_scene_bundle.infos))
            action.text = "Scene reloaded."
            env.step(action)

    def _increase_clutter(self, env: Environment, action: Action) -> None:
        """Adjust the clutter of the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot increase clutter."
            env.step(action)
        else:
            room_id = self.scene_config.room_spec.spec.room_id
            room = self._current_scene_bundle.generator.rooms[room_id]
            current_num_assets = len(room.assets)
            odb = self._current_scene_bundle.generator.odb

            max_floor_objects = int(self._current_scene_bundle.max_floor_objects * 2)
            spawnable_asset_groups = self._current_scene_bundle.spawnable_asset_groups
            spawnable_assets = self._current_scene_bundle.spawnable_assets
            specified_object_types = self._current_scene_bundle.specified_object_types

            asset = None

            for _ in range(max_floor_objects):
                cache_rectangles = asset is None

                if cache_rectangles:
                    rectangle = room.sample_next_rectangle(cache_rectangles=True)
                else:
                    rectangle = room.sample_next_rectangle()

                if rectangle is None:
                    break

                x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(rectangle)

                asset = self._current_scene_bundle.generator.sample_and_add_floor_asset(
                    room=room,
                    rectangle=rectangle,
                    anchor_type=anchor_type,
                    anchor_delta=anchor_delta,
                    spawnable_assets=spawnable_assets,
                    spawnable_asset_groups=spawnable_asset_groups,
                    priority_asset_types = [],
                    odb=odb,
                )

                if asset is None:
                    continue

                # log(f'asset: {asset}')
                room.sample_place_asset_in_rectangle(
                    asset=asset,
                    rectangle=rectangle,
                    anchor_type=anchor_type,
                    x_info=x_info,
                    z_info=z_info,
                    anchor_delta=anchor_delta,
                )

                added_asset_types = []
                if "assetType" in asset:
                    added_asset_types.append(asset["assetType"])
                else:
                    added_asset_types.extend([o["assetType"] for o in asset["objects"]])

                    if not asset["allowDuplicates"]:
                        spawnable_asset_groups = spawnable_asset_groups.query(f"assetGroupName!='{asset['assetGroupName']}'")

                for asset_type in added_asset_types:
                    allow_duplicates_of_asset_type = odb.PLACEMENT_ANNOTATIONS.loc[asset_type.lower()]["multiplePerRoom"]

                    if not allow_duplicates_of_asset_type:
                        # NOTE: Remove all asset groups that have the type
                        spawnable_asset_groups = spawnable_asset_groups[~spawnable_asset_groups[f"has{asset_type.lower()}"]]

                        # NOTE: Remove all standalone assets that have the type
                        spawnable_assets = spawnable_assets[spawnable_assets["assetType"] != asset_type]
        
            self._current_scene_bundle.max_floor_objects = max_floor_objects
            self._current_scene_bundle.spawnable_asset_groups = spawnable_asset_groups
            self._current_scene_bundle.spawnable_assets = spawnable_assets

            object_instances = self._current_scene_bundle.generator.place_other_objects(specified_object_types, current_num_assets)

            max_object_types_per_room = self._current_scene_bundle.max_object_types_per_room * 3
            # all_objects = 
            
            small_object_instances = []
            small_object_instances = add_small_objects(
                object_instances,
                odb,
                self._current_scene_bundle.generator.rooms,
                max_object_types_per_room,
                self._current_scene_bundle.generator.placer.bbox,
                object_counts={},
                specified_object_instances={},
                receptacle_object_counts={}
            )
            new_objects = object_instances + small_object_instances
            print(new_objects)

            self._current_scene_bundle.infos["instances"].extend(new_objects)

            self._locked_scene_bundle = copy.deepcopy(self._current_scene_bundle)
            self._reload_scene(env, action)
        
    def _decrease_clutter(self, env: Environment, action: Action) -> None:
        """Adjust the spacing of the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot decrease spacing."
            env.step(action)
        else:
            pass

    def _increase_spacing(self, env: Environment, action: Action) -> None:
        if not self._locked_scene_bundle:
            action.text = "No locked scene. Cannot increase spacing."
            env.step(action)
        else:
            pass

    def _decrease_spacing(self, env: Environment, action: Action) -> None:
        if not self._locked_scene_bundle:
            action.text = "No locked scene. Cannot decrease spacing."
            env.step(action)
        else:
            pass

    def _analyse(self, env: Environment, action: Action) -> None:
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot analyse"
            env.step(action)
        else:
            scene_config = self.scene_config
            analysis_folder = scene_config.analysis_folder

            save_perspectives(analysis_folder, self._locked_scene_bundle, env, action)

