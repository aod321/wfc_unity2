import sys
sys.path.append("/Users/yinzi/python_projects/AgentEnvCoEvolution")
from gym_wrapper import GymFromDMEnv
import matplotlib.pyplot as plt
import dm_env
import _load_environment as dm_tasks
import einops
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import tensor_utils
from dm_env_rpc.v1 import dm_env_rpc_pb2
from google.protobuf import any_pb2, struct_pb2
import numpy as np
import random
import fastwfc
from PIL import Image, ImageOps

def dm_env_creator_from_port(config, port, host="0.0.0.0"):
    Unity_connection_details = dm_tasks._connect_to_environment(port, host=host,
                    create_world_settings={"seed": config["seed"]},
                    join_world_settings={
                                        "agent_pos_space": config["agent_pos_space"],
                                        "object_pos_space": config["object_pos_space"],
                                        "max_steps": config["max_steps"]
                                        }
                    )
    dm_env = dm_tasks._DemoTasksProcessEnv(Unity_connection_details, config["OBSERVATIONS"], num_action_repeats=config["num_action_repeats"])
    return dm_env

def dm_env_creator_from_local_disk(config):
    settings = dm_tasks.EnvironmentSettings(create_world_settings={"seed": config["seed"]},join_world_settings={
                "agent_pos_space": config["agent_pos_space"],
                "object_pos_space":  config["object_pos_space"],
                "max_steps": config["max_steps"]}, 
                timescale=config["timescale"])
    dm_env = dm_tasks.load_from_disk(config["filename"], settings)
    return dm_env


# Add WFC and gRPC support for unity3D RLLib enviroment
class WFCUnity3DEnv(GymFromDMEnv):
    def __init__(self, env: dm_env.Environment=None, max_steps=2000, wfc_size=9, config=None, file_name=None, port=30051, random_seed=None, host="0.0.0.0"):
        self.set_random_seed(random_seed)
        self.world_name = None
        self.height_map = None
        self.fastwfc = fastwfc.XLandWFC("samples.xml")
        # start from empty aera
        self.wave = self.fastwfc.get_ids_from_wave(self.fastwfc.build_a_open_area_wave())
        self.TASK_OBSERVATIONS = ['RGBA_INTERLEAVED', 'reward', 'done']
        self._SPACE = self.get_space_from_wave(self.wave)
        # all empty tile
        self._SEED = np.ones((wfc_size * wfc_size,1,2)).astype(np.int32)
        self._MAXSTEPS = max_steps
        self.port = port
        self.host = host
        if config is None:
            config = {
                "seed": self._SEED,
                "agent_pos_space": self._SPACE,
                "object_pos_space": self._SPACE,
                "max_steps": self._MAXSTEPS,
                "OBSERVATIONS": self.TASK_OBSERVATIONS,
                "num_action_repeats": 1,
                "filename": file_name,
                "timescale": 1
            }
        else:
            if "wave" in config:
                if config["wave"]:
                    self.set_wave(config["wave"])
            if "seed" not in config:
                config["seed"] = self._SEED
            if "agent_pos_space" not in config:
                config["agent_pos_space"] = self._SPACE
            if "object_pos_space" not in config:
                config["object_pos_space"] = self._SPACE
            if "max_steps" not in config:
                config["max_steps"] = self._MAXSTEPS
            if "OBSERVATIONS" not in config:
                config["OBSERVATIONS"] = self.TASK_OBSERVATIONS
            if "num_action_repeats" not in config:
                config["num_action_repeats"] = 1
            if "filename" not in config:
                config["filename"] = file_name
            if "timescale" not in config:
                config["timescale"] = 1
        
        if env is None:
            if config["filename"] is None:
                 # using port 30051 as default
                if self.port is None:
                    self.port = 30051
                env = dm_env_creator_from_port(config, self.port, host=self.host)
            else:
                env, self.port = dm_env_creator_from_local_disk(config)
        super().__init__(env)

    def get_space_from_wave(self, wave=None):
        # if not wave:
            # wave = self.wave
        # mask , _ = self.PCGWorker_.connectivity_analysis(wave = wave, visualize_ = False, to_file = False)
        # reduce mask to 9x9 for processing
        # reduced_map = einops.reduce(mask,"(h a) (w b) -> h w", a=20, b=20, reduction='max').reshape(-1)
        # use maxium playable area as probility space
        # print(np.flatnonzero(reduced_map == np.argmax(np.bincount(reduced_map))).astype(np.int32))
        # return np.flatnonzero(reduced_map == np.argmax(np.bincount(reduced_map))).astype(np.int32)
        return list(np.arange(81).astype(np.int32))
    
    def set_random_seed(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

    def create_and_join_world(self):
        try:
            connection = dm_tasks._connect_to_environment(self.port, 
                                    create_world_settings={"seed": self._SEED},
                                    join_world_settings={
                                                        "agent_pos_space": self._SPACE,
                                                        "object_pos_space": self._SPACE,
                                                        "max_steps": self._MAXSTEPS
                                                        }
                                    )
            self.connection_details ,self.world_name = connection
            self._env = dm_tasks._DemoTasksProcessEnv(connection, self.TASK_OBSERVATIONS, num_action_repeats=1)
            
        except Exception as e:
                print("Recreate Unity Map World Failed")
                raise e
    
    def extension(self, command="render_map", camera_index=0):
        _EXTENSION_REQUEST = struct_pb2.Struct(fields={"command": struct_pb2.Value(string_value=command), "camera_index": struct_pb2.Value(number_value=camera_index)})
        request = any_pb2.Any()
        request.Pack(_EXTENSION_REQUEST)
        # response = struct_pb2.Struct()
        response = dm_env_rpc_pb2.Tensor()
        success = self._connection.send(request).Unpack(response)
        if success:
            out_tensor = tensor_utils.unpack_tensor(response)
            img = np.array(ImageOps.flip(Image.frombuffer(mode='RGBA', data=out_tensor,size=(512, 512))).convert('RGB'))
            return img
        else:
            return None
        #     dm_env_rpc_pb2.EnvironmentRequest(
        # extension=_wrap_in_any(_EXTENSION_REQUEST)).SerializeToString():
        # dm_env_rpc_pb2.EnvironmentResponse(
        #     extension=_wrap_in_any(_EXTENSION_RESPONSE)),
            
    def reset_world_agent(self, map_seed=None):
        if map_seed is None:
            map_seed = self._SEED
            space = self._SPACE
        else:
            space = self.get_space_from_wave(map_seed)
        # print("reset world and agent")
        self._connection.send(
        dm_env_rpc_pb2.ResetWorldRequest(
            world_name=self._world_name,
            settings={
                "seed": tensor_utils.pack_tensor(map_seed)
            }))

        # rejoin
        self._connection.send(dm_env_rpc_pb2.JoinWorldRequest(world_name=self._world_name, settings={
            "agent_pos_space": tensor_utils.pack_tensor(space),
            "object_pos_space": tensor_utils.pack_tensor(space),
            "max_steps": tensor_utils.pack_tensor(self._MAXSTEPS)
        }))
        self.reset()
   
    def render_in_unity(self, map_seed=None, camera_index=0):
        # self.create_and_join_world()
        self.reset_world_agent()
        return self.extension(camera_index=camera_index)
       
    # conditoinal WFC mutation
    def mutate_a_new_map(self, base_wave=None, weight=162):
        if base_wave is None:
            base_wave = self.wave
        self.wave, _ = self.fastwfc.mutate(base_wave=self.fastwfc.wave_from_id(base_wave), new_weight=weight, iter_count=1, out_img=False)
        self._SPACE = self.get_space_from_wave(self.wave)
        self._SEED = np.array(self.wave).astype(np.int32)
        # 从1开始
        self._SEED[:, 0]+=1
        self._SEED = self._SEED.reshape(-1,1,2)
        return self.wave
    
    def set_wave(self, wave):
        self.wave = wave
        self._SPACE = self.get_space_from_wave(wave)
        self._SEED = np.array(self.wave).astype(np.int32)
        # 从1开始
        self._SEED[:, 0]+=1
        self._SEED = self._SEED.reshape(-1,1,2)
    
    def get_wave(self):
        return self.wave

