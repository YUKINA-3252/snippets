import argparse
import numpy as np
import rospy
from typing import Any, Dict

from eus_imitation_msgs.msg import FloatVector
from imitator.utils.file_utils import (get_config_from_project_name)

class RosManager(object):
    def __init__(self, cfg:Dict[str, Any]):
        self.cfg = cfg
        self.actions_dims = self.cfg.actions.dim
        self.actions = None
        topic_names = self.cfg.actions.topic_name
        self.pub = rospy.Publisher(topic_names, FloatVector, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1 / self.cfg.ros.rate), self._timer_callback)

    def _timer_callback(self, event):
        with open(self.cfg.act_txt_file, 'r') as file:
            full_text = file.read()
            full_text = full_text.replace('\n', ' ')
            lines = full_text.split('] [')
            lines[0] = lines[0].replace('[', '')
            lines[-1] = lines[-1].replace(']', '')
            last_line = lines[-1]
            last_line_list = [float(x) for x in last_line.split()]
            last_line_array = np.array(last_line_list)
            action_msg = FloatVector()
            action_msg.header.stamp = rospy.Time.now()
            action_msg.data = last_line_array
            self.pub.publish(action_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("--act_txt_file", default="pub_act/pub_act.txt")
    args = parser.parse_args()

    config = get_config_from_project_name(args.project_name)
    config.act_txt_file = args.act_txt_file

    rospy.init_node("action_publish_node")

    rm = RosManager(config)

    rospy.spin()
