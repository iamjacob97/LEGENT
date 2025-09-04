from legent import Environment
from legent.scene_generation.objects import get_default_object_db

odb = get_default_object_db()

path_to_executable = "auto" # ".legent/env/client/LEGENT-<platform>-<version>" for example
env = Environment(env_path=path_to_executable, use_animation=False) # or env_path="auto" to start the latest client in .legent/env/client.
try:
    env.reset()
    obs = env.step()
    print(obs.game_states["option_mode_info"])
finally:
    env.close()
