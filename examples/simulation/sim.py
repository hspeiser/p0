from socket import socket, AF_INET, SOCK_DGRAM
from argparse import ArgumentParser
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
    for x in range(-90, 90):
        import time
        time.sleep(0.1)
        client_socket.sendto(json.dumps({"Gripper": x}).encode(), ("localhost", 9999))
    