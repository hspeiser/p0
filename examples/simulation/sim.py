from socket import socket, AF_INET, SOCK_DGRAM
from argparse import ArgumentParser
from lerobot.model.kinematics import RobotKinematics as rk
from lerobot.utils.rotation import Rotation
import numpy as np
joints = ["Pan", "Proximal", "Distal", "Wrist","Roll", "Gripper"]
kinematics = rk("../../rerun_arm/robot.urdf", "jaw_base", joints)
positions = np.array([0, 0, 0, 0, 0])
t_des = np.eye(4, dtype=float)
t_des[:3, :3] = Rotation.from_rotvec(np.array([3.14/2.0, 3.14/2.0, 3.14/2.0],  dtype=float)).as_matrix()
t_des[:3, 3] = np.array([0, 0, 0.2], dtype=float)
thing2 = kinematics.inverse_kinematics(positions, t_des)
print(thing2)
import json
# define command line interface
parser = ArgumentParser()
parser.add_argument('-ip', help='IP')
parser.add_argument('-server_port', type=int, help='Server port')
# parse command line options
args = parser.parse_args()
# create a UDP socket
with socket(AF_INET, SOCK_DGRAM) as client_socket:
    # send message to server
    print(f'Sending message to {"localhost"}:{9999}')
    # for x in range(-90, 90):
    message = {}
    for thing in range(0,6):
        message[joints[thing]] = thing2[thing]
    client_socket.sendto(json.dumps(message).encode(), ("localhost", 9999))
    