from hazalyser import SceneConfig, Controller, HazardRoom, HazardRoomSpec

room_spec = HazardRoomSpec(room_spec_id="ESHA-SingleRoom", 
                           spec=HazardRoom(room_id=1, 
                           room_type="Bedroom"))
scene_config = SceneConfig(room_spec = room_spec, dims = (4, 5), items={"banana": 1},
                           agent_info="Assume the agent is a Tiago robot from PAL Robotics. It is equipped with an omnidirectional drive base, Hokuyo lidar sensors with a 5.6 m range, and two manipulator arms for handling objects and interacting with the subject.", 
                           subject="oldman_ernest.glb", 
                           subject_scale=(1,1,1),
                           subject_info="The subject is an elderly woman who is wheelchair-bound with severely limited mobility, poor vision, and high frailty as measured by the Rockwood Frailty Index. She is highly vulnerable to falls, collisions, and misplacement of objects, and requires careful, low-speed interactions during assistive tasks.", 
                           task="The agent must retrieve a cup of hot water bottle from the table and hand it safely to the subject nearby.", 
                           framework="esha", 
                           llm_key="DSV3", 
                           images="all", 
                           analysis_folder="test")

controller = Controller(scene_config)

controller.start()
