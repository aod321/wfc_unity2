import sys
sys.path.append("../")
from random import random
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
from typing import Callable
from tqdm import tqdm
import os
import gym
import torch
from stable_baselines3 import A2C, PPO, DQN, DDPG
from sb3_contrib import QRDQN
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import fastwfc
from utils import tileid_to_json
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'


MAX_STEPS_PER_EPOSIDE = 5000

def make_env(rank: int) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = WFCUnity3DEnv(wfc_size=9, file_name=gamepath, max_steps=MAX_STEPS_PER_EPOSIDE, random_seed=rank)
        return env
    return _init


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True


if __name__ == "__main__":
    wfc = fastwfc.XLandWFC("samples.xml")
    # Create log dir
    current_time = datetime.now().strftime('%d-%m-%y-%H_%M')
    log_dir = f"./train_logs/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    # ????????????
    gamepath = "/Users/yinzi/Downloads/1126_mac_build_faswfc.app/Contents/MacOS/tilemap_render"
    gamename = "tilemap_render"
    num_env = 3  # Number of env to use
    # ???????????????????????????????????????????????????????????????
    EXTRA_EVAL = True
    TRAIN_EPOSIDES = 2000
    TRAIN_STEPS = 25000
    EVAL_EPOSIDES = 10
    REWARD_THREASHOLD = 0.5
    # Tensroboard log
    tb_logs_path = f"./runs/{current_time}"
    os.makedirs(tb_logs_path, exist_ok=True)
    writer = SummaryWriter(tb_logs_path)
    # ????????????,??????????????????????????????????????????
    print("killing all old processes")
    os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")
    print("Creating all train envs")
    env_list = []
    for i in range(num_env):
        env_list.append(make_env(i))
    vec_env = SubprocVecEnv(env_list)
    vec_env = VecMonitor(vec_env, log_dir)
    print("Creating eval env")
    eval_env = Monitor(WFCUnity3DEnv(wfc_size=9, file_name=gamepath, max_steps=2000))
    base_wave = vec_env.env_method(method_name="get_wave", indices=0)[0]
    eval_env.set_wave(base_wave)
    eval_env.render_in_unity()
    # Create the callback: check every 5000 steps
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    device = torch.device("mps")
    # model = PPO('CnnPolicy', vec_env, verbose=0,  device=device)
    # model = DQN('CnnPolicy', vec_env, verbose=0,  device=device, learning_rate=3e-4,batch_size=512,max_grad_norm=0.5,train_freq=8)
    model = QRDQN('CnnPolicy', vec_env, verbose=0,  device=device)
    sum_evo_count = 0
    map_collections = []
    try:
        print("Evaluation before training: Random Agent")
        # random_agent vecenv??????, before training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
        print(f'Done: Mean reward: {mean_reward} +/- {std_reward:.2f}')
        current_env = 0
        for i in tqdm(range(TRAIN_EPOSIDES)):
            print(f"Training: eposide: {i}/{TRAIN_EPOSIDES} for {TRAIN_STEPS} steps")
            # 1. ??????????????????
            model.learn(total_timesteps=TRAIN_STEPS, callback=save_callback)
            print(f"Training Done for eposide: {i}/{TRAIN_EPOSIDES}, now evaluate for {EVAL_EPOSIDES} eposides")
            # 2. ??????????????????
            eval_env.set_wave(base_wave)
            eval_env.render_in_unity()
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPOSIDES, deterministic=False)
            print(f'Evaluatioon Done: Mean reward: {mean_reward} +/- {std_reward:.2f}')
            writer.add_scalar("eposide_mean_reward", mean_reward, global_step=i)
            # 3. ??????mean_reward??????????????????
            # 3.1 ??????reward >=0.5???????????????????????????????????????,??????????????????
            # 3.2 ??????????????????, ??????????????????????????????????????????
            #-- 3.1
            evolve_count = 0
            while mean_reward > REWARD_THREASHOLD:
                print(f"Current map is too easy for agent now, genrating a new map, count for {evolve_count} times...")
                vec_env.env_method(method_name="mutate_a_new_map", base_wave=base_wave, indices=current_env)
                vec_env.env_method(method_name="render_in_unity", indices=current_env)
                temp_wave = vec_env.env_method(method_name="get_wave", indices=current_env)[0]
                eval_env.set_wave(temp_wave)
                eval_env.render_in_unity()
                print("Saving middle map to json file...")
                filename=os.path.join(log_dir, f'{sum_evo_count}_{evolve_count}_{current_time}_middle.json')
                tileid_to_json(wfc.get_ids_from_wave(temp_wave), save_path=filename)
                print("Evaluating on this new map again ...")
                mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPOSIDES, deterministic=False)
                print(f'Evaluation Done: Mean reward on new map: {mean_reward} +/- {std_reward:.2f}')
                evolve_count +=1
            #-- 3.2
            if evolve_count > 0:
                print(f"{evolve_count} times evolution Done")
                sum_evo_count +=1
                print(f"All Evo times till now: {sum_evo_count}")
                print("Keep current map and continue training")
                # switch to another env window
                if current_env < num_env - 1:
                    current_env += 1
                else:
                    current_env = 0
                # set map to new env
                base_wave = temp_wave
                map_collections.append(temp_wave)
                vec_env.env_method(method_name="set_wave", wave=base_wave, indices=current_env)
                vec_env.env_method(method_name="render_in_unity", indices=current_env)
                print("Save current map to json file...")
                current_time =  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                filename=os.path.join(log_dir, f'{sum_evo_count}_{current_time}.json')
                tileid_to_json(wfc.get_ids_from_wave(base_wave), save_path=filename)
                print("Save current model to file...")
                model.save(os.path.join(log_dir, f"{sum_evo_count}_{current_time}.pth"))
                print("Evaluating on current all map")
                mean_reward_1, std_reward_1 = evaluate_policy(model, vec_env, n_eval_episodes=EVAL_EPOSIDES * num_env, deterministic=False)
                print(f'Evaluation on all Done: Mean reward on all map: {mean_reward} +/- {std_reward:.2f}')
                if EXTRA_EVAL:
                    print("Extra evaluate on all old maps:")
                    extra_rewards_list = []
                    std_rewards_list = []
                    for i, iwave in enumerate(map_collections):
                        print(f"evaluating on map {i}:")
                        eval_env.set_wave(wave=iwave)
                        eval_env.render_in_unity()
                        temp_mean_rewards, temp_std = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=False)
                        std_rewards_list.append(temp_std)
                        extra_rewards_list.append(temp_mean_rewards)
                        writer.add_scalar(f"evo_rewards_{evolve_count}", temp_mean_rewards, global_step=i)
                    for i in range(len(extra_rewards_list)):
                        print(f"mean reward on map{i}: {extra_rewards_list[i]}, std is : {std_rewards_list[i]}")
            else:
                print(f"Continue training without evolution")
    finally:
        current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        print("Save current model to file...")
        model.save(os.path.join(log_dir, "{current_time}_interuppted.pth"))
        vec_env.close()
        print("killing all old processes")
        os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")
                
