"""
File: docker_compose_api.py
---------------------------
This file contains wrapper functions for common docker-compose commands.

"""

import os
from pathlib import Path
import subprocess

DEFAULT_HOST = '127.0.0.1:2375'
DEFAULT_SERVICE = 'cpu'

# TODO: add path to compose file


def start_dockerd(host=DEFAULT_HOST):
    res = subprocess.run('sudo dockerd -H tcp://{} & --log-level error'.format(
                         host), shell=True)


def pull_container(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run('docker-compose --host={} pull {}'.format(host, service),
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   shell=True)


def container_up(host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run('docker-compose --host={} up -d {}'.format(host, service),
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   shell=True)


def container_down(host=DEFAULT_HOST):
    subprocess.run('docker-compose --host={} down'.format(host), check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)


def exec_command_in_container(cmd, host=DEFAULT_HOST, service=DEFAULT_SERVICE):
    subprocess.run('docker-compose --host={} exec {} {}'.format(host, service, 
                   cmd), check=True, shell=True)


def run_command_in_container(cmd, volumes=None, host=DEFAULT_HOST, 
                             service=DEFAULT_SERVICE):
    shell_cmd = ['docker-compose', '--host', host, 'run']
    
    volumes_to_add = []
    cwd = Path.cwd().resolve()
    volumes_to_add.append('-v {}:{}'.format(cwd, cwd))
    for v in volumes:
        if v == '':  # in current directory
            continue

        p = Path(v).resolve()
        volumes_to_add.append('-v {}:{}'.format(p, p))
            
    shell_cmd += volumes_to_add
    shell_cmd += ['-w', str(cwd)]

    shell_cmd += [service, cmd]
    shell_cmd = ' '.join(shell_cmd)
    subprocess.run(shell_cmd, check=True, shell=True)
    
