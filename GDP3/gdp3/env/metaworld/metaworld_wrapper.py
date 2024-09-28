import gym
import numpy as np

import metaworld
import cv2

from termcolor import cprint
from gym import spaces
from gdp3.gym_util.mujoco_point_cloud import PointCloudGenerator
from gdp3.gym_util.mjpc_wrapper import point_cloud_sampling

from gdp3.mask_seg import Mask_Segmentor
from Cutie.dataset_seg import Cutior

from scipy.spatial.transform.rotation import Rotation as R

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.787, 1, -0.4, 100],
    'hammer-v2-goal-observable': [-0.5, -1.5, -0.585, 1, -0.4, 100]
}

class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 num_points=1024,
                 cam_pos = [0.6, 0.265, 0.8]
                 ):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False
        self.cam_pos = cam_pos

        self.env.sim.model.cam_pos[2] = self.cam_pos 
        
        
        #******************************************************
        #This part is for differernt viewpoint experments,
        # cam_pos = self.env.sim.model.cam_pos[2] 
        # cam_quat = self.env.sim.model.cam_quat[2]
        
        
        # cam_Mat = np.eye(4)
        # cam_Mat[:3, 3] = np.array(cam_pos)
        
        # rot = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])
        # rot_mat = rot.as_matrix()
        
        # cam_Mat[:3, :3] = rot_mat
        
        # add_Mat = np.eye(4)
        # add_rot = R.from_euler("xyz", [0, 0, 0], degrees=True)
        # add_mat = add_rot.as_matrix()
        # add_Mat[:3, :3] = add_mat
        # add_Mat[:3, 3] = np.array([0., 0, 0])
        
        # com_Mat = cam_Mat @ add_Mat
        
        # com_rot = R.from_matrix(com_Mat[:3, :3])
        # com_quat = com_rot.as_quat()
        # con_trans = com_Mat[:3, 3]
        
        
        # self.env.sim.model.cam_pos[2] = [con_trans[0], con_trans[1], con_trans[2]]
        # self.env.sim.model.cam_quat[2] = [com_quat[3], com_quat[0], com_quat[1], com_quat[2]]
        #*******************************************************************************************#

        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        self.image_size = 640
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points # 512
        
        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]
    
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            )
        })
        
        self.masked = True
        self.first_seg = True
        self.cap_info = None
        self.track_mask = None
        self.mask_generator = Mask_Segmentor(device)
        self.cutior = Cutior()

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        
        #In the simulation, the left hand coordinate is adopted. For convenience, we transfer all coordinates to right hand
        eef_pos = eef_pos * np.array([-1, 1, 1])
        finger_right = finger_right * np.array([-1, 1, 1])
        finger_left = finger_left * np.array([-1, 1, 1])

        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        return img
    

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
    
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        
        depth = depth[::-1]
        
        return point_cloud, depth
    
    
    def get_reset_mask(self, obs_img, cat_info):
        if self.first_seg:  #manually segmentation
            img = np.uint8(obs_img)#.transpose([1, 2, 0])
            mask, cat_info = self.mask_generator.seg(img)
            self.first_seg = False
        else:      # adjust the video segmentation result
            img = np.uint8(obs_img)#.transpose([1, 2, 0])
            mask = self.cutior.seg_dataset(img)
            mask, cat_info = self.mask_generator.adjust_seg(img, mask, cat_info)
            
            
        self.track_mask = self.cutior.prepar_newseg(img, mask)
        
    
        self.cutior.show_masked_img(img, self.track_mask, cat_info)

        return self.track_mask, cat_info
    
    
    def get_mask(self, obs_img, cat_info):
        img = np.uint8(obs_img)#.transpose([1, 2, 0])
        track_mask = self.cutior.seg_dataset(img)
        
        # t1 = time.time()
       
        # print(time.time() - t1)
        
        self.cutior.show_masked_img(img, track_mask, cat_info)
        cv2.waitKey(1)
        
        return track_mask, cat_info

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        if self.masked:
            mask, cap_info, _ = self.get_mask(obs_pixels, self.cap_info)
            depth, cam_info, cam_pos = self.pc_generator.get_depth()
            point_cloud, kpts = self.pc_generator.generateMaskedPointCloud(obs_pixels, depth, mask, cap_info, cam_info, cam_pos, self.num_points, _, robot_state)
        else:
            point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'kpts': kpts
        }
        return obs_dict
            
    def step(self, action: np.array, show=True):

        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        if self.masked:
            mask, cap_info = self.get_mask(obs_pixels, self.cap_info)
            depth, cam_info, cam_pos = self.pc_generator.get_depth()
            point_cloud = self.pc_generator.generateMaskedPointCloud(obs_pixels, depth, mask, cap_info, cam_info, cam_pos, self.num_points, robot_state)
            
        else:
            point_cloud, depth = self.get_point_cloud()
        # point_cloud = np.array([1])
        # depth = np.array([1])
        if show:
            cv2.imshow("test", obs_pixels)
            cv2.waitKey(1)
        
        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        
        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state
        }

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        
        obs_pixels = self.get_rgb()  
        action = np.ones(4) * 0.01
        raw_obs, reward, done, env_info = self.env.step(action)
        # for i in range(5):
        #     raw_obs, reward, done, env_info = self.env.step(action=np.zeros(4))
        #     time.sleep(0.2)
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        
        if self.masked:
            mask, self.cap_info = self.get_reset_mask(obs_pixels, self.cap_info)
            depth, cam_info, cam_pos = self.pc_generator.get_depth()
            point_cloud= self.pc_generator.generateMaskedPointCloud(obs_pixels, depth, mask, self.cap_info, cam_info, cam_pos, self.num_points, robot_state)
        else:
            point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        # print(point_cloud.shape, "orign pcd shape")
        
        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs
        }
        return obs_dict

    

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass

