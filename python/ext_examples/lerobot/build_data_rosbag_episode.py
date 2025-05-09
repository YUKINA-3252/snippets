import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Sequence, Value
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
import torch
# import torchvision

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy, DiffusionConfig
from lerobot.common.datasets.compute_stats import compute_stats

from build_data_from_rosbag import RosbagEpisode
from pathlib import Path
import pickle

@dataclass
class DummyEpisode:
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    @classmethod
    def create(cls, T: int):
        image_list = []
        state_list = []
        action_list = []
        for _ in range(T):
            rand_image = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
            rand_state = np.random.rand(2)
            rand_action = np.random.rand(2)
            image_list.append(rand_image)
            state_list.append(rand_state)
            action_list.append(rand_action)
        images = np.array(image_list)
        states = np.array(state_list)
        actions = np.array(action_list)
        return cls(images, states, actions)


def convert_to_lerobot_dataset(
    episodes: List[RosbagEpisode],
    fps: int,
    batch_size: int = 32,
    num_workers: int = 8) -> LeRobotDataset:
    # copied and tweaked from
    # https://github.com/ojh6404/imitator/blob/lerobot/imitator/scripts/lerobot_dataset_builder.py

    # create data dict
    ep_dicts = []
    for idx, ep in enumerate(episodes):
        T = len(ep.head_images)
        dones = torch.zeros(T, dtype=torch.bool)
        dones[-1] = True

        ep_dict = {}
        # ep_dict["observation.image"] = [PILImage.fromarray(im) for im in ep.head_images]
        ep_dict["observation.image.head"] = [PILImage.fromarray(im) for im in ep.head_images]
        ep_dict["observation.image.second"] = [PILImage.fromarray(im) for im in ep.second_images]
        ep_dict["observation.state"] = torch.from_numpy(ep.states).float()
        ep_dict["action"] = torch.from_numpy(ep.actions).float()
        ep_dict["episode_index"] = torch.ones(T, dtype=torch.int64) * idx
        ep_dict["frame_index"] = torch.arange(0, T, 1)
        ep_dict["timestamp"] = torch.arange(0, T, 1) / fps
        ep_dict["next.done"] = dones
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    # create haggingface dataset
    dim_state = len(episodes[0].states[0])
    dim_action = len(episodes[0].actions[0])
    features = Features(
        {
            # "observation.image": Image(),
            "observation.image.head": Image(),
            "observation.image.second": Image(),
            "observation.state": Sequence(Value("float32"), length=dim_state),
            "action": Sequence(Value("float32"), length=dim_action),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "timestamp": Value("float32"),
            "next.done": Value("bool"),
            "index": Value("int64"),
        }
    )
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    hf_dataset.set_transform(hf_transform_to_torch)

    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": False,
    }
    lerobot_dataset = LeRobotDataset.from_preloaded(
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
    )
    print("compute stats")
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)
    with open('data/stats.pkl', 'wb') as file:
        pickle.dump(stats, file)
    lerobot_dataset.stats = stats
    return lerobot_dataset


if __name__ == "__main__":
    output_directory = Path("outputs/train/wrapping")
    output_directory.mkdir(parents=True, exist_ok=True)

    # episode_list = []
    # for _ in range(30):
    #     episode_list.append(DummyEpisode.create(80))
    with open('data/rosbag_episode.pkl', 'rb') as file:
        episode_list = pickle.load(file)
    dataset = convert_to_lerobot_dataset(episode_list, 10)
    delta_timestamps = {
        # "observation.image": [-0.1, 0.0],
        "observation.image.head": [-0.1, 0.0],
        "observation.image.second": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset.delta_timestamps = delta_timestamps

    training_steps = 5000
    device = torch.device("cuda")
    log_freq = 250

    cfg = DiffusionConfig(use_separate_rgb_encoder_per_camera=True)
    effective_keys = list(cfg.output_shapes.keys()) + list(cfg.input_shapes.keys()) + ["episode_index", "frame_indx", "index", "next.done",  "timestamp"]
    effective_key_set = set(effective_keys)
    for key, value in dataset.stats.items():
        if key not in effective_key_set:
            continue
        inner_dict = {}
        for key_inner, value_inner in value.items():
            inner_dict[key_inner] = torch.tensor(value_inner)
        dataset.stats[key] = inner_dict

    policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break
    policy.save_pretrained(output_directory)
