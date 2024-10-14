# Learning Generalizable 3D Manipulation Within 10 Demonstrations</a>

Learning robust and generalizable manipulation skills from demonstrations remains a key challenge in robotics, with broad applications in industrial automation and service robotics. While recent imitation learning methods have achieved impressive results, they often require large amounts of demonstration data and struggle to generalize across different spatial variants. In this work, we present a novel framework that learns manipulation skills from as few as 10 demonstrations, yet still generalizes to spatial variants such as different initial object positions and camera viewpoints. Our framework consists of two key modules: Semantic Guided Perception (SGP), which constructs task-focused, spatially aware 3D point cloud
representations from RGB-D inputs; and Spatial Generalized Decision (SGD), an efficient diffusion-based decision-making module that generates actions via denoising. To effectively learn generalization ability from limited data, we introduce a critical spatially equivariant training strategy that captures
the spatial knowledge embedded in expert demonstrations. We validate our framework through extensive experiments on both simulation benchmarks and real-world robotic systems. Our method demonstrates a 60–70% improvement in success rates over state-of-the-art approaches on a series of challenging tasks, even with substantial variations in object poses and camera viewpoints. This work shows significant potential for advancing efficient, generalizable manipulation skill learning in real-world applications.


# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 

# Download Weights

Some pre-trained weights are needed for framework running. Thanks to the advance improvement on founditional model, we can directly use the pre-trained weights.

1.download the pre-trained weights for segment anything1 


- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


and put them under the floder:
`GDP3/SAM/pre_weights/`


2..download the pre-trained weights for Cutie：

- `weight1`: [cutie model 1](https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth)

- `weight2`: [cutie model 2](https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth)

and put them under the floder:
`GDP3/Cutie/weights/`

# 📚 Data
You could generate demonstrations by yourself using our provided expert policies.  Generated demonstrations are under `$YOUR_REPO_PATH/data/`.

# 🛠️ Usage
Scripts for generating demonstrations, training, and evaluation are all provided under GDP3 floder.

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.

For more detailed arguments, please refer to the scripts and the code. We here provide a simple instruction for using the codebase.

1. Generate demonstrations by running "gen_demonstration.sh" under the `$YOUR_REPO_PATH'` as workspace. For example:
    ```bash
    bash GDP3/gen_demonstration.sh soccer 10
    ```
    This will generate 10 demonstrations for the `soccer` task in Metaworld environment. The data will be saved in `data/` folder automatically.

    Once the code starts running, an interactive segmentation window will appear. The first frame will be segmented. Click the left mouse button to add the target object, and right-click to remove background elements if necessary. Press the `space` key to complete segmentation for one object, and press `a` to finish the segmentation process.

    ![Data Collection Animation](data_collection.gif)


2. Train and evaluate a policy with behavior cloning. For example:
    ```bash
    bash GDP3/train_policy.sh gdp3 metaworld_soccer 0112 0 0
    ```
    This will train framework on the `soccer` task in Metaworld environment using point cloud modality. By default we **save** the ckpt (optional in the script).

3. Evaluate a saved policy or use it for inference. For example:
    ```bash
    bash GDP3/eval_policy.sh gdp3 metaworld_soccer 0112 0 0
    ```
    This will evaluate the saved policy that just trained. 



## Results

All result were tested on a single NVIDIA GeForce RTX 4090.

### 1. Simulation Result(Metaworld)


| Method\Task | Dial Turn | Disassemble | Coffee Pull | Soccer | Sweep Into | Hand Insert |
|:------------------:|:--------:|:-----:|:-----:|:-----:|:----------:|:-----------:|
| DP3 | 66.5 | 69.7 | 87.4 | 18.7 | 38.1 | 25.6 |
| Ours | 100.0 | 100.0 | 100.0 | 92.4 | 100.0 | 100.0 |

| Method\Task | Pick Place | Push | Shelf Place| Stick Pull | Pick Place Wall | Button Press |
|:------------------:|:--------:|:-----:|:-----:|:-----:|:----------:|:-----------:|
| DP3 | 56.1 | 51.3 | 48.6 | 27.6 | 45.6 | 100.0 |
| Ours | 100.0 | 95.6 | 95.4 | 89.8 | 100.0 | 100.0 |
<!-- ################################################################## -->
| Method\Task | Button Press | Button Press Topdown | Button Press Wall | Door Close | Coffee Push | Assembly  |
| :---------: | :----------: | :------------------: | :---------------: | :--------: | :---------: | :-------: |
|     DP3     |  100.0   |      100.0       |     99.0      | 100.0  |  94.0   | 99.0  |
|    Ours     |  100.0   |      100.0       |     100.0     |  100.0   | 100.0  | 100.0 |


| Method\Task | Basketball | Bin Picking | Box Close | Door Lock | Drawer Close | Faucet Close |
| :---------: | :--------: | :---------: | :-------: | :-------: | :----------: | :----------: |
|     DP3     | 98.2 | 34.3 | 42.3  | 100.0 |  100.0   |  100.0   |
|    Ours     | 100.0  |     45.5     |    55.0     | 100.0 |  100.0   |  100.0  |

| Method\Task | Faucet Open |  Hammer  | Handle Press | Handle Press Side | Handle Pull | Handle Pull Side |
| :---------: | :---------: | :------: | :----------: | :---------------: | :---------: | :--------------: |
|     DP3     |  100.0  | 76.4 |  100.0   |     100.0     |  53.11  |     85.3     |
|    Ours     |  100.0  |   85.3    |  100.0   |     92.3      |  100.0  |     56.2     |



| Method\Task | Plate Slide Side | Plate Slide | Push Wall | Reach Wall | Stick Push |  Sweep   |
| :---------: | :--------------: | :---------: | :-------: | :--------: | :--------: | :------: |
|     DP3     |    100.0     |  100.0  | 49.8  |  68.3  |  97.4  | 96.3 |
|    Ours     |    100.0     |  100.0  | 100.0 |     80.3     |     97.0      |    97.0    |

| Method\Task | Window Open | Window Close | Drawer Open | Push Back    |
| :---------: | :---------: | :----------: | :---------- | ------------ |
|     DP3     |  100.0  |  100.0   | 100.0   | 0            |
|    Ours     |  100.0  |  100.0   | 100.0       | 0 |

More results will be update as soon as possible




# Acknowledgement
Our code is generally built upon: [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [DexMV](https://github.com/yzqin/dexmv-sim), [DexArt](https://github.com/Kami-code/dexart-release), [VRL3](https://github.com/microsoft/VRL3), [DAPG](https://github.com/aravindr93/hand_dapg), [DexDeform](https://github.com/sizhe-li/DexDeform), [RL3D](https://github.com/YanjieZe/rl3d), [GNFactor](https://github.com/YanjieZe/GNFactor), [H-InDex](https://github.com/YanjieZe/H-InDex), [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [BEE](https://jity16.github.io/BEE/), [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands), [HORA](https://github.com/HaozhiQi/hora). We thank all these authors for their nicely open sourced code and their great contributions to the community.

Contact [Yu Ren](renyu@sia.cn) if you have any questions or suggestions.
