import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'gdp3', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
