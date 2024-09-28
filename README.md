# Learning Generalizable 3D Manipulation Within 10 Demonstrations</a>

Learning robust and generalizable manipulation skills from demonstrations remains a key challenge in robotics, with broad applications in industrial automation and service robotics. While recent imitation learning methods have achieved impressive results, they often require large amounts of demonstration data and struggle to generalize across different spatial variants. In this work, we present a novel framework that learns manipulation skills from as few as 10 demonstrations, yet still generalizes to spatial variants such as different initial object positions and camera viewpoints. Our framework consists of two key modules: Semantic Guided Perception (SGP), which constructs task-focused, spatially aware 3D point cloud
representations from RGB-D inputs; and Spatial Generalized Decision (SGD), an efficient diffusion-based decision-making module that generates actions via denoising. To effectively learn generalization ability from limited data, we introduce a critical spatially equivariant training strategy that captures
the spatial knowledge embedded in expert demonstrations. We validate our framework through extensive experiments on both simulation benchmarks and real-world robotic systems. Our method demonstrates a 60–70% improvement in success rates over state-of-the-art approaches on a series of challenging tasks, even with substantial variations in object poses and camera viewpoints. This work shows significant potential for advancing efficient, generalizable manipulation skill learning in real-world applications.


# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 


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

2. Train and evaluate a policy with behavior cloning. For example:
    ```bash
    bash scripts/train_policy.sh dp3 metaworld_soccer 0112 0 0
    ```
    This will train framework on the `soccer` task in Metaworld environment using point cloud modality. By default we **save** the ckpt (optional in the script).

3. Evaluate a saved policy or use it for inference. For example:
    ```bash
    bash scripts/eval_policy.sh dp3 metaworld_soccer 0112 0 0
    ```
    This will evaluate the saved DP3 policy you just trained. 

