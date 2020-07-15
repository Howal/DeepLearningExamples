# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.distributed as dist

def set_environment_variables_for_nccl_backend(single_node=False):
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']

    if 'AZ_BATCH_MASTER_NODE' in os.environ.keys():
        if not single_node: 
            master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
            os.environ['MASTER_ADDR'] = master_node_params[0]
            os.environ['MASTER_PORT'] = master_node_params[1]
        else:
            os.environ['MASTER_ADDR'] = os.environ['AZ_BATCHAI_MPI_MASTER_NODE']
            os.environ['MASTER_PORT'] = '54965'
        print('NCCL_SOCKET_IFNAME original value = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))
    elif 'OMPI_MCA_orte_hnp_uri' in os.environ.keys():
        # ITP
        master_node_params = os.environ['OMPI_MCA_orte_hnp_uri'].split('tcp://')[1].split(':')
        os.environ['MASTER_ADDR'] = master_node_params[0]
        os.environ['MASTER_PORT'] = '54965'
    else:
        print('no master node info in env')
        exit(1)
    # TODO make this parameterizable
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
    #os.environ['NCCL_IB_DISABLE'] = '0'

    print('RANK = {}'.format(os.environ['RANK']))
    print('WORLD_SIZE = {}'.format(os.environ['WORLD_SIZE']))
    print('MASTER_ADDR = {}'.format(os.environ['MASTER_ADDR']))
    print('MASTER_PORT = {}'.format(os.environ['MASTER_PORT']))
    # print('MASTER_NODE = {}'.format(os.environ['MASTER_NODE']))
    print('NCCL_SOCKET_IFNAME new value = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))

def get_local_rank():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

def get_rank(): # world_rank
    return int(os.environ['OMPI_COMM_WORLD_RANK'])

def get_world_size():
    return int(os.environ['OMPI_COMM_WORLD_SIZE'])

def get_global_size():
    return int(os.environ['OMPI_COMM_WORLD_SIZE'])

def get_local_size():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])	

def is_main_process():
    return get_rank() == 0

def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s
