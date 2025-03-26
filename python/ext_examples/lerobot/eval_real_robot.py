import argparse
import cv2
import glob
import numpy as np
import os
from pathlib import Path
import pickle
import socket
import struct
import torch
from typing import Any, Dict, List, Optional, Type

from imitator.utils.file_utils import (get_config_from_project_name,
                                       get_models_dir)

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionConfig, DiffusionPolicy

from safetensors.torch import load_file

def rename_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # "observation_image" -> "observation_image_head"
        if "buffer_observation_image." in key:
            print("###############")
            new_key_head = key.replace("buffer_observation_image.", "buffer_observation_image_head.")
            new_key_second = key.replace("buffer_observation_image.", "buffer_observation_image_second.")
            new_state_dict[new_key_head] = value
            new_state_dict[new_key_second] = value
        else:
            new_key = key
    return new_state_dict

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

    def get_image(self, camera_type):
        jpg_files = glob.glob(os.path.join("sub_obs", camera_type, '*jpg'))
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

    def get_data(self):
        with socket.socket(socket.AR_INET, socket.SOCKET_STREAM) as s:
            s.bind(('localhost', 65432))
            s.listen()
            conn, addr = s.accept()
            with conn:
                data_size = conn.recv(8)
                data_size = struct.unpack('>Q', data_size)[0]
                data = b''
                while len(data) < data_size:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet

                received_data = pickle.loads(data)
                return received_data

    def feedback(self, policy: DiffusionPolicy, n_pixel=96):
        while True:
            # data = self.get_data()
            head_rgb = self.get_image("head_images")
            second_rgb = self.get_image("second_images")
            # rgb = data['image']
            head_image = torch.from_numpy(head_rgb).float()
            second_image = torch.from_numpy(second_rgb).float()
            head_image = head_image.to(torch.float32) / 255
            second_image = second_image.to(torch.float32) / 255
            head_image = head_image.permute(2, 0, 1)
            second_image = second_image.permute(2, 0, 1)
            head_image = head_image.unsqueeze(0)
            second_image = second_image.unsqueeze(0)

            state = torch.from_numpy(np.array(self.get_obs())).float().unsqueeze(0)
            # state = torch.from_numpy(np.array(data['float_list']).float().unsqueeze(0)
            head_image = head_image.to("cuda")
            second_image = second_image.to("cuda")
            state = state.to("cuda")

            observation = {
                # "observation.image": head_image,
                "observation.image.head": head_image,
                "observation.image.second": second_image,
                "observation.state": state
                }
            with torch.inference_mode():
                action = policy.select_action(observation)
            action_np = action.detach().cpu().numpy().flatten()
            with open("pub_act/pub_act.txt", 'a') as file:
                file.write(f'{action_np}\n')
            # done = termiennated | truncated | done
            # print(done)
            print(action_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--feedback", action="store_tr", help="feedback mode")
    parser.add_argument("-pn", type=str, default="wrapping", help="project name")
    parser.add_argument("--head_img_dir", default="sub_obs/head_images")
    parser.add_argument("--second_img_dir", default="sub_obs/second_images")
    parser.add_argument("--obs_txt_file", default="sub_obs/obs.txt")
    parser.add_argument("-pp", type=str, help="project path name.")
    parser.add_argument("-n", type=int, default=100, help="epoch num")
    parser.add_argument("-m", type=int, default=112, help="pixel num")
    parser.add_argument("-model", type=str, default="lstm", help="select prop model if --feedback specified")
    parser.add_argument("-untouch", type=int, default=5, help="num of untouch episode")
    parser.add_argument("--model_dir", type=str, default="outputs/train/wrapping")
    args = parser.parse_args()
    n_epoch: int = args.n
    n_pixel: int = args.m
    # feedback_mode: bool = args.feedback
    project_name: str = args.pn
    n_untouch: int = args.untouch
    project_path_str: Optional[str] = args.pp
    model_str: str = args.model
    model_dir: str = args.model_dir

    config = get_config_from_project_name(args.pn)
    config.obs_txt_file = args.obs_txt_file

    # remove txt file
    with open("pub_act/pub_act.txt", 'w') as file:
        pass

    feedback_mode = True
    if feedback_mode:
        om = ObsManager(config)
        with Path("./data/stats.pkl").open("rb") as f:
            stats = pickle.load(f)
        for key, value in stats.items():
            for key_sub, value_sub in value.items():
                stats[key][key_sub] = value_sub.to("cuda")
        print(stats)

        cfg = DiffusionConfig(use_separate_rgb_encoder_per_camera=True)
        effective_keys = list(cfg.output_shapes.keys()) + list(cfg.input_shapes.keys()) + ["episode_index", "frame_indx", "index", "next.done",  "timestamp"]
        effective_key_set = set(effective_keys)
        for key, value in stats.items():
            if key not in effective_key_set:
                continue
            inner_dict = {}
            for key_inner, value_inner in value.items():
                inner_dict[key_inner] = torch.tensor(value_inner)
        stats[key] = inner_dict

        policy = DiffusionPolicy(cfg, dataset_stats=stats)
        policy = policy.to("cuda")
        # print("#######################")
        # print(torch.load(os.path.join(model_dir, "model.pth")).keys())
        # print("###########################")
        # print(policy.state_dict().keys())
        # pretrained_weights = torch.load(os.path.join(model_dir, "model.pth"))
        pretrained_weights = load_file(os.path.join(model_dir, "model.safetensors"))
        # fixed_weights = rename_keys(pretrained_weights)
        policy.load_state_dict(pretrained_weights)

        om.feedback(policy, n_pixel)
