import os
import zarr
import cv2

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import json
import tqdm
import time


class Cutior(object):
    def __init__(self) -> None:
        self.cutie = get_default_model()
        # Typically, use one InferenceCore per video
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.mask_color = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
                      np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255])]
        pass
        
    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def seg_dataset(self, img):

        image = Image.fromarray(np.uint8(img))
        image = to_tensor(image).cuda().float()


        # otherwise, we propagate the mask from memory
        output_prob = self.processor.step(image)


        # convert output probabilities to an object mask
        mask = self.processor.output_prob_to_mask(output_prob)
        mask_array = mask.cpu().numpy().astype(np.uint8)

            
        return mask_array

    def prepar_newseg(self, img, mask):

        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        
        self.mask = Image.fromarray(np.uint8(mask))       
        objects = np.unique(np.array(self.mask))
        # background "0" does not count as an object
        objects = objects[objects != 0].tolist()

        # mask = self.mask
        # assert mask.mode in ['L', 'P']
        # palette = mask.getpalette()
        # objects = np.unique(np.array(self.mask))
        # background "0" does not count as an object
        # objects = objects[objects != 0].tolist()

        mask = torch.from_numpy(np.array(mask)).cuda()
        
        
        image = Image.fromarray(np.uint8(img))
        image = to_tensor(image).cuda().float()
        
        
        
        output_prob = self.processor.step(image, mask, objects=objects)
        
        mask = self.processor.output_prob_to_mask(output_prob)
            
            
        mask_array = mask.cpu().numpy().astype(np.uint8)
        
        return mask_array
        
    def show_masked_img(self,img, mask, cap_info):
        
        orign_img = img
        mask = mask
        
        color_mask = np.zeros(orign_img.shape, dtype=np.uint8)
        for i, (k,v) in enumerate(cap_info.items()):
            color_mask[mask == v] = self.mask_color[i]
        
        alpha = 0.5
        meta = 1 - alpha
        gamma = 0
        image = cv2.addWeighted(orign_img, alpha, color_mask, meta, gamma)
        
        
        cv2.imshow("masked_img", image)
        cv2.waitKey(1)
        

class Data_processer(object):
    def __init__(self, data_pth, output_pth) -> None:
        self.data_pth = data_pth
        self.output_pth = output_pth
        self.root = zarr.open(self.data_pth, mode = "r")
        self.sample_group = self.root['data']
        # self.sam = sam_model_registry["vit_h"](checkpoint="/data/ubuntu_data/Code/Robot_Diffusion/FastSAM/SAM/sam_vit_h_4b8939.pth")
        # self.predictor = SamPredictor(self.sam)
        # self.sam.to(device="cuda:0")
        
        self.pts_list = []
        self.pts_label = []
        self.mask_input = None
        
        
        
        self.cutie = get_default_model()
        # Typically, use one InferenceCore per video
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        
        pass
    
    
    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def seg_dataset(self):
        # mask = Image.fromarray(self.mask.astype(np.uint8))
        mask = self.mask
        assert mask.mode in ['L', 'P']
        palette = mask.getpalette()
        objects = np.unique(np.array(self.mask))
        # background "0" does not count as an object
        objects = objects[objects != 0].tolist()

        mask = torch.from_numpy(np.array(mask)).cuda()
        
        images = self.sample_group['img'][:]
        
        for ti in tqdm.trange(images.shape[0]):
        # load the image as RGB; normalization is done within the model
            img = images[ti]
            image = Image.fromarray(np.uint8(img))
            image = to_tensor(image).cuda().float()

            if ti == 0:
                # if mask is passed in, it is memorized
                # if not all objects are specified, we propagate the unspecified objects using memory
                output_prob = self.processor.step(image, mask, objects=objects)
            else:
                # otherwise, we propagate the mask from memory
                t1 = time.time()
                output_prob = self.processor.step(image)
                t2 = time.time()
                print(t2 -t1)

            # convert output probabilities to an object mask
            mask = self.processor.output_prob_to_mask(output_prob)
            mask_array = mask.cpu().numpy().astype(np.uint8)
            
            img_name = str(ti) + ".png"
            img_pth = os.path.join(self.output_pth, img_name)
            cv2.imwrite(img_pth, mask_array)
            
            objects = np.unique(np.array(mask_array))
            objects = objects[objects != 0].tolist()
            # visualize prediction
            # mask = Image.fromarray(mask_array)
            # mask.putpalette(palette)
            # mask.show()  # or use mask.save(...) to save it somewhere
            # cv2.imshow("mask", mask_array)
            
        
        return 1
    
    def load_mask(self, mask_pth):
        
        json_pth = os.path.join(mask_pth, "cap.json")
        with open(json_pth, 'r') as f:
            data = json.load(f)
        
        jpg_pth = os.path.join(mask_pth, "mask_overall.png")
        mask_overall = cv2.imread(jpg_pth)
        mask_overall = np.sum(mask_overall, axis=2, dtype=np.uint8)

        self.mask = Image.fromarray(mask_overall)       
        objects = np.unique(np.array(self.mask))
        # background "0" does not count as an object
        objects = objects[objects != 0].tolist()
        # self.mask = self.mask.convert("P")
        # test_mask_pth = "/data/ubuntu_data/Code/Robot_Diffusion/Cutie/examples/masks/bike/00000.png"
        # test_mask = cv2.imread(test_mask_pth)
        t = 1
        
        
        
    
def main():
    data_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/metaworld_soccer_expert.zarr"
    out_pth = "/data/ubuntu_data/Code/Robot_Diffusion/3D-Diffusion-Policy/3D-Diffusion-Policy/data/metaworld_soccer_expert.zarr/seg/masks"
    data_processer = Data_processer(data_pth, out_pth)
    
    img = data_processer.sample_group['img'][:]
    first_img = img[0]
    
    mask_pth = os.path.join(data_pth, "seg")
    data_processer.load_mask(mask_pth)
    data_processer.seg_dataset()
    
    # seg_res = data_processer.seg(first_img)
    
    
    
    
    
    
    
    i = 0
    
    
    
if __name__ == "__main__":
    main()