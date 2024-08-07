import argparse
import cv2
import glob
import numpy as np
import os
from pathlib import Path
import pickle
import torch
from typing import Any, Dict, List, Optional, Type

from imitator.utils.file_utils import (get_config_from_project_name,
                                       get_models_dir)

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

class ObsManager(object):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.obs_keys = list(self.cfg.obs.keys())
        self.obs_dims = {
            key: self.cfg.obs[key].dim for key in self.obs_keys
        }  # obs_key : dim
        self.obs = {
            obs_key: None for obs_key in self.obs_keys
        }  # initialize obs with None

        topic_names = [self.cfg.obs[key].topic_name for key in self.obs_keys]

    def get_image(self):
        jpg_files = glob.glob(os.path.join(self.cfg.img_dir, '*jpg'))
        jpg_files.sort(key=os.path.getctime, reverse=True)
        # latest_file = max(jpg_files, key=os.path.getmtime)
        second_latest_file = jpg_files[1]
        image = cv2.imread(second_latest_file)
        return image

    def get_obs(self):
        with open(self.cfg.obs_txt_file, 'r') as file:
            full_text = file.read()
            full_text = full_text.replace('\n', ' ')
            lines = full_text.split('] [')
            lines[0] = lines[0].replace('[', '')
            lines[-1] = lines[-1].replace(']', '')
            last_line = lines[-1]
            last_line_list = [float(x) for x in last_line.split()]
            last_line_array = np.array(last_line_list)
            return last_line_array

    def feedback(self, policy: DiffusionPolicy, n_pixel=112):
        rgb_list = []
        while True:
            rgb = self.get_image()
            rgb_list.append(rgb)

            if rgb.size == 0:
                return
            image = torch.from_numpy(rgb).float()
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            state = torch.from_numpy(np.array(self.get_obs())).float().unsqueeze(0)
            image = image.to("cuda")
            state = state.to("cuda")

            observation = {
                "observation.image": image,
                "observation.state": state
                }
            with torch.inference_mode():
                action = policy.select_action(observation)
            action_np = action.detach().cpu().numpy().flatten()
            with open(self.cfg.action_txt_file, 'a') as file:
                file.write(f'{action_np}\n')
            # print(action_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", action="store_true", help="feedback mode")
    parser.add_argument("-pn", type=str, default="wrapping", help="project name")
    parser.add_argument("--img_dir", default="sub_obs/sub_img")
    parser.add_argument("--obs_txt_file", default="sub_obs/obs.txt")
    parser.add_argument("--action_txt_file", default="pub_act/pub_act.txt")
    parser.add_argument("-pp", type=str, help="project path name.")
    parser.add_argument("-n", type=int, default=100, help="epoch num")
    parser.add_argument("-m", type=int, default=112, help="pixel num")
    parser.add_argument("-model", type=str, default="lstm", help="select prop model if --feedback specified")
    parser.add_argument("-untouch", type=int, default=5, help="num of untouch episode")
    parser.add_argument("--model_dir", type=str, default="outputs/train/wrapping")
    args = parser.parse_args()
    n_epoch: int = args.n
    n_pixel: int = args.m
    feedback_mode: bool = args.feedback
    project_name: str = args.pn
    n_untouch: int = args.untouch
    project_path_str: Optional[str] = args.pp
    model_str: str = args.model
    model_dir: str = args.model_dir

    config = get_config_from_project_name(args.pn)
    config.img_dir = args.img_dir
    config.obs_txt_file = args.obs_txt_file
    config.action_txt_file = args.action_txt_file

    # remove txt file
    with open(config.action_txt_file, 'w') as file:
        pass

    if feedback_mode:
        om = ObsManager(config)
        with Path("./data/stats.pkl").open("rb") as f:
            stats = pickle.load(f)
        for key, value in stats.items():
            for key_sub, value_sub in value.items():
                stats[key][key_sub] = value_sub.to("cuda")
        policy = DiffusionPolicy(dataset_stats=stats)
        policy = policy.to("cuda")
        policy.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))

        om.feedback(policy, n_pixel)
