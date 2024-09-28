from typing import Dict
import torch
import numpy as np
import copy
from gdp3.common.pytorch_util import dict_apply
from gdp3.common.replay_buffer import ReplayBuffer
from gdp3.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from gdp3.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gdp3.dataset.base_dataset import BaseDataset

class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            obs_step=2,
            rot_aug=True
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            keys=['state', 'action', 'point_cloud'],
            episode_mask=train_mask, zarr_pth=zarr_path)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.zarr_path = zarr_path
        self.rot_aug = rot_aug
        self.obs_step = obs_step

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            keys=['state', 'action', 'point_cloud'],
            episode_mask=~self.train_mask,zarr_pth=self.zarr_path
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud']
            
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def generate_random_aug(self):
        from scipy.spatial.transform.rotation import Rotation as R
        z_angle = np.random.random([1]) * 360
        aug_rot = R.from_euler("xyz", [0, 0, z_angle], degrees=True)
        aug_rmat = aug_rot.as_matrix()
        aug_Mat = np.eye(4)
        aug_Mat[:3, :3] = aug_rmat
        return aug_Mat

    def aug_pointcloud(self, pointcloud, aug_Mat):
        aug_pc = []
        pts = pointcloud[:, :, :3]
        for i in range(pts.shape[0]):
            aug_pt = (aug_Mat[:3, :3] @ pts[i].T).T
            aug_pc.append(aug_pt)
        
        aug_pts = np.stack(aug_pc, axis=0)
        
        aug_pointcloud = copy.deepcopy(pointcloud)
        aug_pointcloud[:, :, :3] = aug_pts
        
        return aug_pointcloud

    def aug_agent_pos(self, agent_pos, aug_Mat, obs_step):
        cur_obs_id = obs_step - 1
        # ap_dim = agent_pos.shape
        
        # axis_trans = np.array([-1, 1, 1])  # transfer the agent pos left coor 2 right coor
        # axis_trans_ = np.concatenate([axis_trans, axis_trans, axis_trans])
        
        cur_pos = agent_pos[cur_obs_id][:3]
        N = agent_pos.shape[0]
        D = agent_pos.shape[1]
        
        
        o_apos = copy.deepcopy(agent_pos)
        o_apos = o_apos.reshape([-1, 3])

        central_apos = o_apos - cur_pos
        
        aug_center_ap = (aug_Mat[:3, :3] @ central_apos.T).T
        aug_ap = aug_center_ap + cur_pos + np.array([np.random.random(1)[0]-0.5,np.random.random(1)[0]-0.5 , 0])
        
        aug_ap = aug_ap.reshape([N, D])
             
        return aug_ap

    def aug_actions(self, actions, aug_Mat):
        aug_actions_list = []
        for i in range(actions.shape[0]):
            action = actions[i, :3].reshape([1, 3])
            ag_action = (aug_Mat[:3, :3] @ action.T).T #@ aug_Mat.T
            aug_actions_list.append(ag_action)
            
        aug_actions_array = np.concatenate(aug_actions_list, axis=0)
        
        aug_actions = copy.deepcopy(actions)
        aug_actions[:, :3] = aug_actions_array
        
        return aug_actions

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        
        
        N = agent_pos.shape[0]
        D = agent_pos.shape[1]
        agent_pos = agent_pos.reshape([-1, 3])
        agent_pos = agent_pos * np.array([-1, 1, 1])
        agent_pos = agent_pos.reshape([N, D])
        
                
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        
        
        actions = sample['action'].astype(np.float32)
        
        actions[:, 0] = actions[:, 0] * (-1)
        
        
        if self.rot_aug:
            
            aug_Mat = self.generate_random_aug()
            
            point_cloud = self.aug_pointcloud(point_cloud, aug_Mat)
            agent_pos = self.aug_agent_pos(agent_pos, aug_Mat, self.obs_step)
            actions = self.aug_actions(actions, aug_Mat)
            
            
            

            
            
        data = {
            'obs': {
                'point_cloud': point_cloud, 
                'agent_pos': agent_pos
                
            },
            'action': actions #sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence_wkpts(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

