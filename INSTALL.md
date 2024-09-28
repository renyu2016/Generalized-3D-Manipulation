# Installing Conda Environment 

The following guidance works well for a machine with 4090 GPU, cuda 12.1, driver 535.183.01 .

First, git clone this repo and `cd` into it.

    git clone https://github.com/renyu2016/Generalized-3D-Manipulation.git


1.create python/pytorch env

    conda remove -n gdp3 --all
    conda create -n gdp3 python=3.8
    conda activate gdp3


---

2.install torch

    # if using cuda>=12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version

---

3.install mujoco in `~/.mujoco`

    cd ~/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

    tar -xvzf mujoco210.tar.gz

and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl


and then install mujoco-py (in the folder of `third_party`):

    cd YOUR_PATH_TO_THIRD_PARTY
    cd mujoco-py-2.1.2.14
    pip install -e .
    cd ../..


----

4.install sim env

    pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

    cd third_party
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..
---

5.install pytorch3d (a simplified version)

    cd third_party/pytorch3d_simplified && pip install -e . && cd ..

---
6.install some necessary packages
    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor

7.install SAM 
    cd `GDP3/SAM`
    pip install -e .

8.install Cutie
    cd `GDP3/Cutie`
    pip install -e .

9.For visualization, the `opencv-contrib-python==4.1.2.30` is recommanded:
    pip install opencv-contrib-python==4.1.2.30


