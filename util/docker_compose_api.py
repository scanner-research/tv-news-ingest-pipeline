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
    res = subprocess.run('sudo dockerd -H tcp://{} & --log-level error'.format(host),
                         shell=True)


def pull_container(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run('DOCKER_HOST={} docker-compose pull {}'.format(host, service),
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)


def container_up(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run('DOCKER_HOST={} docker-compose up -d {}'.format(host, service),
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)


def container_down(host=DEFAULT_HOST):
    subprocess.run('DOCKER_HOST={} docker-compose down'.format(host), check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)


def run_command_in_container(cmd, host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run('DOCKER_HOST={} docker-compose exec {} {}'.format(host, service, cmd),
                   check=True, shell=True)
