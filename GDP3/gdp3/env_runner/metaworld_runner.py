import wandb
import numpy as np
import torch
import collections
import tqdm
from gdp3.env import MetaWorldEnv
from gdp3.gym_util.multistep_wrapper import MultiStepWrapper
from gdp3.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from gdp3.policy.base_policy import BasePolicy
from gdp3.common.pytorch_util import dict_apply
from gdp3.env_runner.base_runner import BaseRunner
import gdp3.common.logger_util as logger_util
from termcolor import cprint

import os
import cv2
import json
import time
import copy

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 cam_pos = [0.6, 0.265, 0.8]
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points, cam_pos=cam_pos)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        from data.check_data import Mask_Segmentor
        import sys
        sys.path.append('/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/third_party')
        from Cutie.dataset_seg import Cutior
        
        self.first_seg = True 
        
        
        record_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs"
        self.traj_file = os.path.join(record_pth, "path.npy")
        self.succ_info = os.path.join(record_pth, "succ_info.json")
        # self.mask_generator = Mask_Segmentor()
        # self.cutior = Cutior()
        
    def get_fisrt_pcd(self, obs):
        if self.first_seg:  #manually segmentation
            img = obs['image']
            img = np.uint8(img).transpose([1, 2, 0])
            mask, self.cat_info = self.mask_generator.seg(img)
            self.first_seg = False
        else:      # adjust the video segmentation result
            img = obs['image']
            img = np.uint8(img).transpose([1, 2, 0])

            track_mask = self.cutior.seg_dataset(img)
            mask, self.cat_info = self.mask_generator.adjust_seg(img, track_mask, self.cat_info)
            
        self.track_mask = self.cutior.prepar_newseg(img, mask)

        obs_depth = obs['depth']
        cam_info = obs['cam_info']
        cam_pos = obs['cam_pos']
        pcd_array = self.env.pc_generator.generateMaskedPointCloud(img, obs_depth, self.track_mask, self.cat_info, cam_info, cam_pos)
        print(pcd_array.shape)
        return pcd_array
        
        
        
        
        
    # def get_masked_pcd(self, obs):
    #     obs_img = obs['image']
    #     img = np.uint8(obs_img).transpose([1, 2, 0])
    #     self.track_mask = self.cutior.seg_dataset(img)
    #     self.cutior.show_masked_img(img, self.track_mask, self.cat_info)
        
    #     obs_depth = obs['depth']
    #     cam_info = obs['cam_info']
    #     cam_pos = obs['cam_pos']
    #     pcd_array = self.env.pc_generator.generateMaskedPointCloud(img, obs_depth, self.track_mask, self.cat_info, cam_info, cam_pos)
        
    #     print(pcd_array.shape)
    #     return pcd_array
        

        

    # def mask_run(self, policy: BasePolicy, save_video=False):
    #     device = policy.device
    #     dtype = policy.dtype

    #     all_traj_rewards = []
    #     all_success_rates = []
    #     env = self.env

        
    #     for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Mask Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
    #         # start rollout
    #         obs = env.reset()
    #         policy.reset()

    #         done = False
    #         traj_reward = 0
    #         is_success = False
            
    #         obs['point_cloud'] = self.get_fisrt_pcd(obs)
    #         obs.pop('cam_info')
    #         obs.pop('cam_pos')
        
                
            
    #         # i = 0
    #         # s_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/run_images"
    #         while not done:
    #             np_obs_dict = dict(obs)
    #             obs_dict = dict_apply(np_obs_dict,
    #                                   lambda x: torch.from_numpy(x).to(
    #                                       device=device))

    #             with torch.no_grad():
    #                 obs_dict_input = {}
    #                 obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
    #                 obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
    #                 print(obs_dict_input['point_cloud'].shape, "masked pcd shape")
    #                 action_dict = policy.predict_action(obs_dict_input)

    #             np_action_dict = dict_apply(action_dict,
    #                                         lambda x: x.detach().to('cpu').numpy())
    #             action = np_action_dict['action'].squeeze(0)

    #             obs, reward, done, info = env.mask_step(action)
                
    #             obs['point_cloud'] = self.get_masked_pcd(obs)
    #             obs.pop('cam_info')
    #             obs.pop('cam_pos')
                
    #             traj_reward += reward
    #             done = np.all(done)
    #             is_success = is_success or max(info['success'])

    #         all_success_rates.append(is_success)
    #         all_traj_rewards.append(traj_reward)
            

    #     max_rewards = collections.defaultdict(list)
    #     log_data = dict()

    #     log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
    #     log_data['mean_success_rates'] = np.mean(all_success_rates)

    #     log_data['test_mean_score'] = np.mean(all_success_rates)
        
    #     cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

    #     self.logger_util_test.record(np.mean(all_success_rates))
    #     self.logger_util_test10.record(np.mean(all_success_rates))
    #     log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
    #     log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

    #     videos = env.env.get_video()
    #     if len(videos.shape) == 5:
    #         videos = videos[:, 0]  # select first frame
        
    #     if save_video:
    #         print("save_video")
    #         videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
    #         log_data[f'sim_video_eval'] = videos_wandb

    #     _ = env.reset()
    #     videos = None

    #     return log_data


    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env
        
        
        obs_soccer_pos = []
        traj_info = []
        npy_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/eval_soccer.npy"
        len_info_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/eval_soccer.json"
	
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            try:
                obs = env.reset()
            except:
                print(f"{episode_idx} episode something wrong when reset the environment!!")
                continue
                pass
            
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            i = 0
            s_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/run_images"
            while not done:
                # cur_obs_id = self.n_obs_steps - 1
                # cur_pos = obs['agent_pos'][cur_obs_id][:3]
                # N = obs['agent_pos'].shape[0]
                # D = obs['agent_pos'].shape[1]
                
                
                # o_apos = copy.deepcopy(obs['agent_pos'])
                # o_apos = o_apos.reshape([-1, 3])

                # central_apos = o_apos - cur_pos
                # obs['agent_pos'] = central_apos.reshape([N, D])
                # pre_t1 = time.time()
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0) 
                action[:, 0] = action[:, 0] * (-1)
                
                
                # print("predict time:", time.time()- pre_t1)
                
                # print(action)
                
                obs, reward, done, info = env.step(action)
                
                # print(info['obj_pos'][0])
                # obs_soccer_pos.append(info['obj_pos'][0])

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
                if is_success:
                    break

            all_success_rates.append(is_success)
            print(all_success_rates)
            all_traj_rewards.append(traj_reward)
            
            traj_info.append([len(obs_soccer_pos), is_success])
           
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        
        # obs_soccer_pos = np.concatenate(obs_soccer_pos)
        # np.save(npy_pth, obs_soccer_pos)

        with open(len_info_pth, 'w') as f:
            json.dump({"succ_info":traj_info}, f)
        

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            print("save_video")
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb
        try:
            _ = env.reset()
        except:
            print("eval process finish, but something was wrong with reset env")
        videos = None

        return log_data
