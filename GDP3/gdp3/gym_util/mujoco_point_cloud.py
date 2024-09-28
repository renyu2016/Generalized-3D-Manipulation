# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d
import cv2
import os

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""



def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, cam_names:List, img_size=84):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_names = cam_names
        
        # List of camera intrinsic matrices
        self.cam_mats = []
        
        for idx in range(len(self.cam_names)):
            # get camera id
            cam_id = self.sim.model.camera_name2id(self.cam_names[idx])
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, save_img_dir=None, device_id=0):
        o3d_clouds = []
        cam_poses = []
        depths = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            color_img, depth = self.captureImage(self.cam_names[cam_i], capture_depth=True, device_id=device_id)
            depths.append(depth)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                self.saveImg(depth, save_img_dir, "depth_test_" + str(cam_i))
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth)
            
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
            
            # od_color = o3d.geometry.Image(color_img)  # Convert the color image to Open3D format                
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(od_color, od_depth)  # Create an RGBD image
            # o3d_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd_image,
            #     od_cammat)

            # Compute world to camera transformation matrix
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        # get numpy array of point cloud, (position, color)
        combined_cloud_points = np.asarray(combined_cloud.points)
        # color is automatically normalized to [0,1] by open3d
        

        # combined_cloud_colors = np.asarray(combined_cloud.colors)  # Get the colors, ranging [0,1].
        combined_cloud_colors = color_img.reshape(-1, 3) # range [0, 255]
        combined_cloud = np.concatenate((combined_cloud_points, combined_cloud_colors), axis=1)
        depths = np.array(depths).squeeze()
        return combined_cloud, depths


    def get_depth(self, device_id=0):
        depths = []
        camera_infos = []
        camera_pos = []
        for cam_i in range(len(self.cam_names)):
            color_img, depth = self.captureImage(self.cam_names[cam_i], capture_depth=True, device_id=device_id)
            # depth = self.verticalFlip(depth)
            depths.append(depth)
            
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
           
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            
            camera_infos.append(od_cammat)
            camera_pos.append(c2w)
            
            
            
        
        depths = np.array(depths).squeeze()
        # camera_infos = np.array(camera_infos).squeeze()
        # camera_pos = np.array(camera_pos).squeeze()
        
        
        return depths, camera_infos, camera_pos

    def process_pcd(self, point_cloud, points_num, cap_info):
        x_min, y_min, z_min, x_max, y_max, z_max= [-0.9, -0.9, -0.9, 0.9, 0.9, 0.1]
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        if True:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
        
        labs = point_cloud[:, 3]
        
        cat_nums = 0
        for k,v in cap_info.items():
            cur_ids = np.where(labs == v)
            if len(cur_ids[0]) == 0:
                continue
            cat_nums += 1
        
        per_pnum = points_num // cat_nums
        
        s_pointcloud = []
        for k,v in cap_info.items():
            if v == 0:
                s_nums = per_pnum + points_num % cat_nums
            else:
                s_nums = per_pnum
                
            cur_ids = np.where(labs == v)
            if len(cur_ids[0]) == 0:
                continue
            sampled_indices = np.random.choice(len(cur_ids[0]), s_nums, replace=True)
            s_ids = cur_ids[0][sampled_indices]
            s_pts = point_cloud[s_ids, :]
            s_pointcloud.append(s_pts)
            
        point_cloud = np.concatenate(s_pointcloud, axis=0)

        return point_cloud


    def generateMaskedPointCloud(self, rgb, depth, mask, cap_info, cam_info, cam_pos, points_num, gripper_pos):
        # mask[mask>0] = 255
        depth = self.verticalFlip(depth)
        

        masked_Rgb = np.ones(rgb.shape) * (-1)
        for k,v in cap_info.items():
            masked_Rgb[mask==v] = np.array([v, -1, -1])
        
        # for k,v in cap_info.items():
        #     masked_Rgb[k_mask==v] = np.array([-1, v, -1])
        
        od_depth = o3d.geometry.Image(np.ascontiguousarray(depth))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(od_depth, cam_info[0])
        
        
        
        cam_pos = self.sim.model.cam_pos[2]
        cam_quat = self.sim.model.cam_quat[2]
    

        cam_Mat1 = np.eye(4)
        cam_Mat1[:3, :3] = quat2Mat(cam_quat)
        cam_Mat1[:3, 3] = cam_pos
        
        cam_initmat = np.eye(4)
        cam_initmat[:3, :3] = np.array([[-1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, -1]])
        
        cor_trans = np.eye(4)
        cor_trans[0, 0] = -1    # transform from left coordinate to right coordinates
        
        pts = np.array(pcd.points)
        pts[:, :3] = (pts[:, :3] * np.array([-1, 1, 1]))
        
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.transform(cor_trans @ cam_Mat1 @ cam_initmat)
        
        g_pos = np.array([gripper_pos[0], gripper_pos[1], gripper_pos[2]])
        
        pcd = pcd.translate(g_pos * (-1))
        
        od_rgb = masked_Rgb.reshape([-1, 3])
        pcd.colors = o3d.utility.Vector3dVector(od_rgb)
        
        pts_array = np.array(pcd.points)
        cls_array = np.array(pcd.colors)
        pcd_array = np.concatenate([pts_array, cls_array], axis=1)
        
        pcd_array = self.process_pcd(pcd_array, points_num, cap_info)
        
        return pcd_array
     
    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True, device_id=0):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth, device_id=device_id)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)

            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)
            return img, depth_convert
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")
