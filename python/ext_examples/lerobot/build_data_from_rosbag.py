import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Sequence, Value
import torch
import torchvision

import argparse
from Crypto.Cipher import AES
import csv
from cv_bridge import CvBridge
from eus_imitation_msgs.msg import FloatVector
import eus_imitation_utils.ros_utils as RosUtils
from functools import partial
import imitator.utils.file_utils as FileUtils
import message_filters
import os
import pickle
import rosbag
import rospy
from sensor_msgs.msg import CompressedImage, Image, JointState

@dataclass
class DummyEpisode:
    images: np.ndarray
    states: np.ndarray
    # actions: np.ndarray

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

@dataclass
class RosbagEpisode:
    head_images: np.ndarray
    second_images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    action_buf = []
    obs_buf = dict()
    # config: dict
    # ts: dict

    # def __init__(self, images, states, actions, config):
    #     self.images = images
    #     self.states = states
    #     self.actions = actions
    #     self.config = config

    @classmethod
    def callback(cls, config, *msgs):
        for topic, msg in zip(config['topics'], msgs):
            if topic == config['action_topic']:  # action
                data = np.array(msg.data).astype(np.float32)
                config['action_buf'].append(data)
                RosbagEpisode.action_buf.append(data)
            else:
                if "Image" in msg._type:
                    if "Compressed" in msg._type:
                        data = config['img_bridge'].compressed_imgmsg_to_cv2(msg, "rgb8").astype(
                            np.uint8
                        )
                    else:
                        data = config['img_bridge'].imgmsg_to_cv2(msg, "rgb8").astype(np.uint8)
                    data = cv2.resize(
                        data, tuple(config['obs_cfg'][config['topics_to_keys'][topic]].dim[:2]), interpolation=cv2.INTER_LANCZOS4
                    )
                elif "JointState" in msg._type:
                    data = np.array(
                        [
                            msg.position[msg.name.index(joint)]
                            for joint in config['obs_cfg'][config['topics_to_keys'][topic]].joints
                        ]
                    ).astype(np.float32)
                else:
                    data = np.array(msg.data).astype(np.float32)
                RosbagEpisode.obs_buf[config['topics_to_keys'][topic]].append(data)

    @classmethod
    def create(cls, bag, config):
        print(bag)
        # get dataset config
        mf_cfg = config.ros.message_filters
        obs_cfg = config.obs
        action_cfg = config.actions

        obs_keys = list(obs_cfg.keys())
        obs_topics = [obs.topic_name for obs in obs_cfg.values()]
        obs_msg_types = [eval(obs.msg_type) for obs in obs_cfg.values()]
        action_topic = action_cfg.topic_name
        action_msg_type = eval(action_cfg.msg_type)

        topics_to_keys = dict(zip(obs_topics + [action_topic], obs_keys + ["action"]))
        topics = obs_topics + [action_topic]
        msg_types = obs_msg_types + [action_msg_type]

        primary_image_key = None
        for obs_key in obs_keys:
            if obs_cfg[obs_key].get("camera") == "primary":
                primary_image_key = obs_key
                break

        subscribers = dict()
        for topic, msg_type in zip(topics, msg_types):
            subscribers[topic] = message_filters.Subscriber(topic, msg_type)

        img_bridge = CvBridge()

        config_dict = {
            'topics': topics,
            'action_topic': action_topic,
            'action_buf': [],
            'obs_buf': {key: [] for key in obs_keys},
            'img_bridge': img_bridge,
            'obs_cfg': obs_cfg,
            'topics_to_keys': topics_to_keys,
            'subscribers': subscribers,
            'actions': config.actions
            }

        ts = message_filters.ApproximateTimeSynchronizer(
            subscribers.values(),
            queue_size=mf_cfg.queue_size,
            slop=mf_cfg.slop,
            allow_headerless=False,
        )

        ts.registerCallback(partial(cls.callback, config_dict))
        # obs_buf = dict()
        RosbagEpisode.action_buf = []
        for obs_key in obs_keys:
            RosbagEpisode.obs_buf[obs_key] = []

        bag_reader = rosbag.Bag(bag, skip_index=True)

        # get action and obs buffer
        for message_idx, (topic, msg, t) in enumerate(
                bag_reader.read_messages(topics=topics)
                ):
            subscriber = subscribers.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        # action
        if config.actions.type == "action_trajectory":
            action_data = np.array(RosbagEpisode.action_buf)
        elif config.actions.type == "proprio_trajectory":
            action_data = np.diff(np.array(obs_buf["proprio"]), axis=0)
            #repeat last action
            action_data = np.concatenate(
                [action_data, action_data[-1:]], axis=0
                )
        else:
            raise NotImplementedError
        actions = np.array(action_data)

        states = np.array([])
        # obs
        for obs_key, obs_data in RosbagEpisode.obs_buf.items():
            obs_data = np.array(obs_data)
            if obs_key == 'head_image':
                head_images = np.array(obs_data)
            if obs_key == 'second_image':
                second_images = np.array(obs_data)
            if obs_key == 'robot_state':
                states = np.array(obs_data)
        return cls(head_images, second_images, states, actions)

if __name__ == "__main__":
    rospy.Time = RosUtils.PatchTimer

    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--rosbag_dir", type=str)
    args = parser.parse_args()

    config = FileUtils.get_config_from_project_name(args.project_name)

    if args.rosbag_dir is None:
        args.rosbag_dir = os.path.join(
            FileUtils.get_data_dir(args.project_name), "rosbags"
        )
    rosbags = RosUtils.get_rosbag_abs_paths(args.rosbag_dir)
    print("Found {} rosbags".format(len(rosbags)))

    episode_list = []
    # for _ in range(30):
    #     episode_list.append(DummyEpisode.create(80))
    for _, bag in enumerate(rosbags):
        episode_list.append(RosbagEpisode.create(bag, config))

    # output
    with open('data/rosbag_episode.pkl', 'wb') as file:
        pickle.dump(episode_list, file)
