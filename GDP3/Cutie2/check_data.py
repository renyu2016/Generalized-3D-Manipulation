import os
import numpy
import cv2
import zarr
import open3d as o3d

from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import copy




        
        

class Data_processer(object):
    def __init__(self, data_pth, output_pth) -> None:
        self.data_pth = data_pth
        self.output_pth = output_pth
        self.root = zarr.open(self.data_pth, mode = "r")
        self.sample_group = self.root['data']
        self.sam = sam_model_registry["vit_h"](checkpoint="/data/ubuntu_data/Code/Robot_Diffusion/FastSAM/SAM/sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device="cuda:0")
        
        self.pts_list = []
        self.pts_label = []
        self.mask_input = None
        
        pass
    
    
    def mouse(self, event, x, y, flags, param):
        img = param
        mask_input = param
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            self.pts_list.append(np.array([x,y]))
            self.pts_label.append(1)
            
            # cv2.circle(img, (x, y), 1, (255, 255, 255), thickness = -1)
            # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
            #             1.0, (255, 255, 255), thickness = 1)
            
            input_point = np.array(self.pts_list) #np.array([[389, 527]])
            input_label = np.array(self.pts_label)
            mask_input = self.mask_input
            print("add click and segment")
            masks, scores, logits  = self.predictor.predict(point_coords=input_point, point_labels=input_label, mask_input=mask_input)
            
            best_idx = np.argmax(scores)
            self.best_mask = masks[best_idx, :, :]
            self.mask_input = logits[best_idx, :, :].reshape([1, logits.shape[1], logits.shape[2]])
            
            seg_img = np.zeros_like(img)
            seg_img[self.best_mask] = img[self.best_mask]
            
            cv2.imshow("image2", seg_img)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            self.pts_list.append(np.array([x,y]))
            self.pts_label.append(0)
            
            input_point = np.array(self.pts_list) #np.array([[389, 527]])
            input_label = np.array(self.pts_label)
            mask_input = self.mask_input
            print("mine click and segment")
            masks, scores, logits  = self.predictor.predict(point_coords=input_point, point_labels=input_label, mask_input=mask_input)
            
            best_idx = np.argmax(scores)
            self.best_mask = masks[best_idx, :, :]
            self.mask_input = logits[best_idx, :, :].reshape([1, logits.shape[1], logits.shape[2]])
            
            seg_img = np.zeros_like(img)
            seg_img[self.best_mask] = img[self.best_mask]
            
            cv2.imshow("image2", seg_img)
    
    
    def get_cpt(self, input_img):
        self.pts_list = []
        self.pts_label = []
        self.mask_input = None
        self.best_mask = None
        
        img = copy.deepcopy(input_img)
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", self.mouse, img)

        cv2.waitKey(0)
        cv2.putText(img, "specify the obj_name", (int(img.shape[0]/2), int(img.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness = 1)
        cv2.imshow("image", img)
        cat = input("Input the category name:")
        
        cv2.destroyWindow("image")
        cv2.destroyWindow("image2")
        
        return {cat:self.best_mask} 
        

    def seg(self, img):
        # img = self.sample_group['img'][:]
        obs_img = copy.deepcopy(img)
        self.predictor.set_image(obs_img)
        seg_finish = False
        self.seg_res = {}
        
        while not seg_finish:
            res_img = np.zeros_like(obs_img)
            seg_res = self.get_cpt(obs_img)
            for key, value in seg_res.items():
                self.seg_res[key] = value
            for key, value in self.seg_res.items():
                res_img[value] = obs_img[value]
                idx = np.where(value > 0)
                cv2.putText(res_img, key, (idx[0][1], idx[0][0]), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness = 1)
                
            
            
            cv2.imshow("main_win", res_img)
            key = cv2.waitKey()
            if chr(key) == "a":
                seg_finish = True
        
        return self.seg_res
    
    
    
    
    
    
    def fastseg(self):
        model = FastSAM('weights/FastSAM-s.pt')
        IMAGE_PATH = './images/dogs.jpg'
        DEVICE = 'cuda:1'
        
        img = self.sample_group['img'][:]
        obs_img = img[0]
        
        
        
        everything_results = model(obs_img, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(obs_img, everything_results, device=DEVICE)

        # everything prompt
        # ann = prompt_process.everything_prompt()
        # ann = prompt_process.text_prompt(text='a photo of a shelf') 
        ann = prompt_process.point_prompt(points=[[451, 551]], pointlabel=[1]) # 37 70

        prompt_process.plot(annotations=ann,output_path='./output/obs_img.jpg',)
                
        return 1
    
    def show_data(self):
        img = self.sample_group['img'][:]
        pcd = self.sample_group['point_cloud']
        
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window("pcd")
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        g_pcd = o3d.geometry.PointCloud()
        pts = pcd[0, :, :3]
        # cls = pcd[0, :, 3:]
        g_pcd.points = o3d.utility.Vector3dVector(pts)
        # g_pcd.colors = o3d.utility.Vector3dVector(cls)
        visualizer.add_geometry(frame)
        visualizer.add_geometry(g_pcd)
        data_len = img.shape[0]
        for i in range(data_len):
            obs_img = img[i]
            cv2.imshow("obs_img", obs_img)
            cv2.waitKey(1)
            
            pcd_v = pcd[i]
            pts = pcd_v[:, :3]
            # cls = pcd_v[:, 3:]
            g_pcd.points = o3d.utility.Vector3dVector(pts)
            # g_pcd.colors = o3d.utility.Vector3dVector(cls)
            visualizer.update_geometry(g_pcd)
            visualizer.poll_events()
            visualizer.update_renderer()
            
            
            
        return 1

def main():
    data_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/metaworld_shelf-place_expert.zarr"
    out_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/p_metaworld_shelf-place_expert.zarr"
    data_processer = Data_processer(data_pth, out_pth)
    
    img = data_processer.sample_group['img'][:]
    first_img = img[0]
    seg_res = data_processer.seg(first_img)
    
    
    
    
    
    
    i = 0
    
    
#     root = zarr.open(data_pth, mode = "r")
#     sample_group = root['data']
#     img = sample_group['img'][:]
#     pcd = sample_group['point_cloud']
    
#     visualizer = o3d.visualization.Visualizer()
#     visualizer.create_window("pcd")
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#     g_pcd = o3d.geometry.PointCloud()
#     pts = pcd[0, :, :3]
#     # cls = pcd[0, :, 3:]
#     g_pcd.points = o3d.utility.Vector3dVector(pts)
#     # g_pcd.colors = o3d.utility.Vector3dVector(cls)
#     visualizer.add_geometry(frame)
#     visualizer.add_geometry(g_pcd)
#     data_len = img.shape[0]
#     for i in range(data_len):
#         obs_img = img[i]
#         cv2.imshow("obs_img", obs_img)
#         cv2.waitKey(1)
        
#         pcd_v = pcd[i]
#         pts = pcd_v[:, :3]
#         # cls = pcd_v[:, 3:]
#         g_pcd.points = o3d.utility.Vector3dVector(pts)
#         # g_pcd.colors = o3d.utility.Vector3dVector(cls)
#         visualizer.update_geometry(g_pcd)
#         visualizer.poll_events()
#         visualizer.update_renderer()
    


# key_list = 1


if __name__ == "__main__":
    main()