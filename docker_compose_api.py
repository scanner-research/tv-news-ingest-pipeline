"""
File: docker_compose_api.py
---------------------------
This file contains wrapper functions for common docker-compose commands.

"""

import subprocess

DEFAULT_HOST = '127.0.0.1:2375'
DEFAULT_SERVICE = 'cpu'

# TODO: add path to compose file


def start_dockerd(host=DEFAULT_HOST):
    res = subprocess.run(f'sudo dockerd -H tcp://{host} & --log-level error',
                         shell=True)


def pull_container(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run(f'DOCKER_HOST={host} docker-compose pull cpu', check=True,
                   capture_output=True, shell=True)


def container_up(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run(f'DOCKER_HOST={host} docker-compose up -d {service}',
                   check=True, capture_output=True, shell=True)


def container_down(host=DEFAULT_HOST):
    subprocess.run(f'DOCKER_HOST={host} docker-compose down', check=True,
                   capture_output=True, shell=True)


def run_command_in_container(cmd, host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run(f'DOCKER_HOST={host} docker-compose exec {service} {cmd}',
                   check=True, shell=True)
