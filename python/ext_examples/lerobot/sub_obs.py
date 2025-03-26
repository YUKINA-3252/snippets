import argparse
from copy import deepcopy
import cv2
from cv_bridge import CvBridge
from datetime import datetime
from functools import partial
import glob
import numpy as np
import os
import pickle
import rospy
from sensor_msgs.msg import CompressedImage, Image, JointState
import socket
import struct
from typing import Any, Dict

from eus_imitation_msgs.msg import FloatVector
from imitator.utils.file_utils import (get_config_from_project_name,
                                       get_models_dir)

data = {
    'head_image': None,
    'second_image': None,
    'float_list': []
    }

class RosManager(object):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.obs_keys = list(self.cfg.obs.keys())
        self.obs_dims = {
            key: self.cfg.obs[key].dim for key in self.obs_keys
        }  # obs_key : dim
        self.obs = {
            obs_key: None for obs_key in self.obs_keys
        }  # initialize obs with None

        self.bridge = CvBridge()
        self.sub_obs = [
            rospy.Subscriber(
                self.cfg.obs[key].topic_name,
                eval(self.cfg.obs[key].msg_type),
                partial(self._obs_callback, key),
                queue_size=1,
                buff_size=2**24,
            )
            for key in self.obs_keys
        ]

        topic_names = [self.cfg.obs[key].topic_name for key in self.obs_keys]
        rospy.loginfo("Subscribing to topics: {}".format(topic_names))

        self.timer = rospy.Timer(rospy.Duration(1 / self.cfg.ros.rate), self._timer_callback)

    def _obs_callback(self, obs_key, msg):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.obs[obs_key].msg_type == "CompressedImage":
            self.obs[obs_key] = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.obs[obs_key] = cv2.resize(self.obs[obs_key], (self.cfg.obs[obs_key].dim[0], self.cfg.obs[obs_key].dim[1]))
            if self.cfg.obs[obs_key].topic_name == "/head_camera/rgb/image_raw/compressed":
                data['head_image'] = self.obs[obs_key]
                serialized_data = pickle.dumps(data)
                save_path = os.path.join("sub_obs", "head_images", f'{current_time}.jpg')
            else:
                data['second_image'] = self.obs[obs_key]
                serialized_data = pickle.dumps(data)
                save_path = os.path.join("sub_obs", "second_images", f'{current_time}.jpg')
            cv2.imwrite(save_path, self.obs[obs_key])

            # self.obs[obs_key] = ros_numpy.numpify(msg)
        elif self.cfg.obs[obs_key].msg_type == "JointState":
            if len(msg.position) != 0:
                self.obs[obs_key] = np.array(
                    [
                        msg.position[msg.name.index(joint)]
                        for joint in self.cfg.obs[obs_key].joints
                    ]
                ).astype(np.float32)
        elif self.cfg.obs[obs_key].msg_type == "FloatVector":
            self.obs[obs_key] = np.array(msg.data).astype(np.float32)
            # data['float_list'] = self.obs[obs_key]
            # seralized_data = pickle.dumps(data)
            with open("sub_obs/obs.txt", 'a') as file:
                file.write(f'{self.obs[obs_key]}\n')
        else:
            raise NotImplementedError(f"msg_type {self.cfg.obs[obs_key].msg_type} not supported")

        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     s.connect(("127.0.0.1", 50000))
        #     s.sendall(struct.pack('>Q', len(serialized_data)))
        #     s.sendall(serialized_data)


    def get_image(self):
        for obs_key in self.obs_keys:
            obs_msg = rospy.wait_for_message(
                self.cfg.obs[obs_key].topic_name,
                eval(self.cfg.obs[obs_key].msg_type),
                timeout=5.0,
            )
            self._obs_callback(obs_key, obs_msg)

    def _timer_callback(self, event):
        """
        Timer callback for real-world environments.
        """
        self.obs_buf = deepcopy(self.obs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("--img_dir", default="sub_obs/sub_img")
    parser.add_argument("--obs_txt_file", default="sub_obs/obs.txt")
    args = parser.parse_args()

    config = get_config_from_project_name(args.project_name)
    config.img_dir = args.img_dir
    config.obs_txt_file = args.obs_txt_file

    # remove jpg file
    head_img_jpg_files = glob.glob(os.path.join("sub_obs/head_images", '*.jpg'))
    for file in head_img_jpg_files:
        if os.path.isfile(file):
            os.remove(file)
            print(f'Removed file: {file}')
    second_img_jpg_files = glob.glob(os.path.join("sub_obs/second_images", '*.jpg'))
    for file in second_img_jpg_files:
        if os.path.isfile(file):
            os.remove(file)
            print(f'Removed file: {file}')
    # remove txt file
    with open(config.obs_txt_file, 'w') as file:
        pass

    rospy.init_node("eval_node")

    rm = RosManager(config)
    rm.get_image()

    rospy.spin()
