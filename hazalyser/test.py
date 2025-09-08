from hazalyser import SceneConfig, SceneGenerator, Controller

scene_config = SceneConfig(subject="person_in_a_wheelchair.glb")
scene_generator = SceneGenerator(scene_config)

controller = Controller(scene_generator)

controller.start()

