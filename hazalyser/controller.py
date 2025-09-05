import re
from typing import Dict, List, Optional, Tuple, Any

from hazalyser.utils import deepcopy_scene
from hazalyser.generator import SceneGenerator
from hazalyser.helpers import get_current_scene_state
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

    def __init__(self, scene_generator: SceneGenerator = SceneGenerator()) -> None:
        self.scene_generator: SceneGenerator = scene_generator
        self._locked_scene: Optional[Dict[str, Any]] = None
        self._current_scene: Optional[Dict[str, Any]] = None
        self._running: bool = False
        self.marked_scenes = []
        self.command_handlers = {
            "#LOCK": self._lock_scene,
            "#UNLOCK": self._unlock_scene,
            "#RESET": self._reset_scene,
            "#COMMIT": self._commit_scene,
            "#RELOAD": self._reload_scene,
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
                        self.marked_scenes.append(self._current_scene)
                        marked = True
                        env.step(action)

                elif cmd == "#LIST":
                    if self._locked_scene:
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
                    if not self._locked_scene:
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
        scene = self.scene_generator.generate()
        self._current_scene = scene
        env.reset(ResetInfo(scene=scene))    

    def _show_scene(self, env: Environment, action: Action, index: int) -> None:
        """Helper to show a specific scene."""
        scene_num = index % len(self.marked_scenes)
        scene = self.marked_scenes[scene_num]
        self._current_scene = scene
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
                if self._locked_scene:
                    action.text = "Scene locked. Cannot go to next."
                    env.step(action)
                else:
                    current_index += 1
                    self._show_scene(env, action, current_index)

            elif cmd == "#PREV":
                if self._locked_scene:
                    action.text = "Scene locked. Cannot go to previous."
                    env.step(action)
                else:
                    current_index -= 1
                    self._show_scene(env, action, current_index)
            
            elif cmd.startswith("#SCENE"):
                scene_num = re.search(r"\d+", cmd)
                if scene_num:
                    if self._locked_scene:
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
                if self._locked_scene:
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
        env.reset(ResetInfo(scene=self._current_scene))
        action.text = "Scene reset."
        env.step(action)
        
    def _lock_scene(self, env: Environment, action: Action) -> None:
        self._locked_scene = deepcopy_scene(self._current_scene)
        action.text = "Scene locked."
        env.step(action)

    def _unlock_scene(self, env: Environment, action: Action) -> None:
        self._locked_scene = None
        action.text = "Scene unlocked."
        env.step(action)

    def _commit_scene(self, env: Environment, action: Action) -> None:
        """Save current scene state of the locked scene"""
        if not self._locked_scene:
            action.text = "No scene locked. Cannot save."
            env.step(action)
        else:
            current_state = get_current_scene_state(env)
            obs_instances = current_state["instances"]
            locked_instances = self._locked_scene["instances"]
            print(locked_instances, "\n")

            self._locked_scene["player"]["position"] = current_state["player"]["position"]
            self._locked_scene["player"]["rotation"] = current_state["player"]["rotation"]
            self._locked_scene["agent"]["position"] = current_state["agent"]["position"]
            self._locked_scene["agent"]["rotation"] = current_state["agent"]["rotation"]
            
            num_objects = len(obs_instances)
            for obj_id in range(num_objects):
                if locked_instances[obj_id]["prefab"] in obs_instances[obj_id]["prefab"]:
                    self._locked_scene["instances"][obj_id]["position"] = obs_instances[obj_id]["position"]
                    self._locked_scene["instances"][obj_id]["rotation"] = obs_instances[obj_id]["rotation"]

            print(self._locked_scene["instances"])
            action.text = "Scene saved."
            env.step(action)
    
    def  _reload_scene(self, env: Environment, action: Action) -> None:
        if not self._locked_scene:
            action.text = "No scene locked. Cannot reload."
            env.step(action)
        else:
            env.reset(ResetInfo(scene=self._locked_scene))
            action.text = "Scene reloaded."
            env.step(action)