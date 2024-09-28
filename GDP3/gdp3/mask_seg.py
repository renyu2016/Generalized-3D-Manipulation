import os
import numpy
import cv2
import zarr
import open3d as o3d
import visualizer

from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import copy
import json
import sys
import tqdm

from scipy.spatial.transform.rotation import Rotation as R


class Mask_Segmentor(object):
    def __init__(self, device) -> None:
        self.device = device
        self.sam = sam_model_registry["vit_h"](checkpoint="GDP3/SAM/pre_weights/sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device=self.device)
        
        self.mask_color = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
                      np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255])]
        pass

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
                cv2.destroyWindow("main_win")
        return self.output_res(self.seg_res)

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
    
    
    def get_cpt(self, input_img, cat=None):
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
        if cat is None:
            cat = input("Input the category name:")
        
        cv2.destroyWindow("image")
        cv2.destroyWindow("image2")
        
        return {cat:self.best_mask} 

    def check_segres(self, img, track_mask, cap_info):
        
        mask_color = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
                      np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255])]
        
        orign_img = img
        mask = track_mask
        
        color_mask = np.zeros(orign_img.shape, dtype=np.uint8)
        for i, (k,v) in enumerate(cap_info.items()):
            color_mask[mask == v] = mask_color[i]
        
        alpha = 0.5
        meta = 1 - alpha
        gamma = 0
        image = cv2.addWeighted(orign_img, alpha, color_mask, meta, gamma)
        key_select = "'a' for continue / 'n' for maually segment"
        cv2.putText(image, key_select, (np.uint8(image.shape[0]/2), np.uint8(image.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness = 1)

        cv2.imshow("masked_img", image)
        input_key = cv2.waitKey(3000)
        return input_key

    def output_res(self, seg_res):
        label_captation = {"0": 0}
        mask_value = []
        for i, (key, value) in enumerate(seg_res.items()):
            # repeat = True
            # while repeat:
            #     val = np.random.randint(0, 255)
            #     repeat = val in mask_value
            # val_list = [val - j for j in range(6)] + [val + j for j in range(6)]
            # mask_value += val_list
            c_v = i + 1
            try:
                mask_overall[value] = c_v
            except:
                mask_overall = np.zeros([value.shape[0], value.shape[1]])
                mask_overall[value] = c_v
            label_captation[key] = c_v
        
        return (mask_overall, label_captation)

    def refine_seg(self, img, track_mask, cap_info):
        track_mask = np.zeros(track_mask.shape)
        self.predictor.set_image(img)
        for k,v in cap_info.items():
            if k == '0':
                continue
            print("refine the mask of", k)
            seg_res = self.get_cpt(img, k)
            track_mask[seg_res[k]] = v
        
        return track_mask, cap_info

    def adjust_seg(self, img, track_mask, cap_info):
        input_key = self.check_segres(img, track_mask, cap_info)
        
        if input_key == -1:
            return track_mask, cap_info
        
        if chr(input_key) == 'a':
            print("success_seg")
            return track_mask, cap_info
        elif chr(input_key) == 'n':
            print("re segmentation")
            
            mask = self.refine_seg(img, track_mask, cap_info)
            return mask
          
  
          
    
  
    
    

    

   