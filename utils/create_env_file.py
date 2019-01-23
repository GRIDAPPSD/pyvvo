#!/usr/bin/python3
"""Create file pyvvo/.env for setting environment variables at runtime.

Assuming Python version is >=3.5

In PyCharm, using the EnvFile plugin to set environment variables:
https://github.com/Ashald/EnvFile
"""
import socket
import json
import argparse


def main(platform, out_file):
    """Create environment.json file for use with PyCharm EnvFile plugin.
    """
    # Initialize dictionary to hold environment variables.
    env_dict = dict()
    # Get host's IP address.
    env_dict['host_ip'] = get_ip()

    # Whether or not we're running inside the platform.
    env_dict['platform'] = platform

    # Write to file.
    with open(out_file, 'w') as f:
        json.dump(env_dict, f)


def get_ip():
    """This method returns the "primary" IP on the local box
        (the one with a default route).

    - Does NOT need routable net access or any connection at all.
    - Works even if all interfaces are unplugged from the network.
    - Does NOT need or even try to get anywhere else.
    - Works with NAT, public, private, external, and internal IP's
    - Pure Python 2 (or 3) with no external dependencies.
    - Works on Linux, Windows, and OSX.

    Source: Jamieson Becker:
    https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # noinspection PyBroadException
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="File to write environment variables to.",
                        default="/home/thay838/git/pyvvo/pyvvo/env.json")
    parser.add_argument("--platform",
                        help=("1/0, whether or not pyvvo is being run inside "
                              + "the GridAPPS-D platform via docker-compose"),
                        default="1")
    parser.add_argument("--port", "-p", help="Port for GridAPPS-D API.",
                        default="61613")
    args = parser.parse_args()
    # Check inputs.
    if not isinstance(args.file, str):
        raise TypeError('--file input must be a string.')

    if not isinstance(args.platform, str):
        raise TypeError('--platform input must be a string.')

    if not (args.platform == "1" or args.platform == "0"):
        raise ValueError('--platform must be either a 0 or a 1.')

    main(platform=args.platform, out_file=args.file)
