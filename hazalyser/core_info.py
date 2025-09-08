from legent import Environment
from legent.action.action import ResetInfo
from legent.scene_generation.objects import get_default_object_db
from hazalyser.helpers import get_mesh_size

odb = get_default_object_db()

path_to_executable = "auto" # ".legent/env/client/LEGENT-<platform>-<version>" for example
subject_path = "D:\\Dissertation\\LEGENT_REPO\\LEGENT\\hazalyser\\obsAssets\\subject\\oldman_ernest.glb"
size = get_mesh_size(subject_path)
env = Environment(env_path=path_to_executable, use_animation=False) # or env_path="auto" to start the latest client in .legent/env/client.
try:
    
    scene = {
        "instances": [
            {
                "prefab": "LowPolyInterior2_Bin",
                "position": [0,0.1,2],
                "rotation": [0, 90, 0],
                "scale": [1, 1, 1],
                "type": "interactable"
            },
            {
                "prefab": "LowPolyInterior_Potato",
                "position": [0,0.1,0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "type": "interactable"
            },
            {
                "prefab": "LowPolyInterior_Floor_01",
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [4, 1, 4],
                "type": "kinematic"
            },
            {
                "prefab": subject_path,
                "position": [0, size[1]/2, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "type": "kinematic"
            }
            
        ],
        "player": {
            "position": [0,0.1,1],
            "rotation": [0, 180, 0]
        },
        "agent": {
            "position": [0,0.1,-1],
            "rotation": [0, 0, 0]
        },
        "center": [0, 10, 0],
        "prompt": ""
    }
    obs = env.reset(ResetInfo(scene=scene))
    print(scene)
    print(obs.game_states)
    while True:
        env.step()
finally:
    env.close()
