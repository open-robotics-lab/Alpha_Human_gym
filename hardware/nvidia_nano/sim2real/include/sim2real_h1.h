/*
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2025 Darren Wang. All rights reserved.
*/
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <thread>
#include "enumClass.h"
#include <iostream>
#include "stdio.h"

struct _msg_request
{
    float trigger;
    float command[4];
    float eu_ang[3];
    float omega[3];
    float acc[3];
    float q[10];
    float dq[10];
    float tau[10];
    float init_pos[10];
};

struct _msg_response
{
    float q_exp[10];
    float dq_exp[10];
    float tau_exp[10];
};

class RL_H1_UDP {
public:
    std::string model_path;
    void init_policy();
    void load_policy();
    torch::Tensor model_infer(torch::Tensor policy_input);
    void handleMessage(_msg_request request);

    //gamepad
    float smooth = 0.03;
    float dead_zone = 0.01;

    float cmd_x = 0.;
    float cmd_y = 0.;
    float cmd_rate = 0.;

    std::vector<float> action;
    std::vector<float> action_temp;
    std::vector<float> prev_action;

    torch::Tensor action_buf;
    torch::Tensor obs_buf;
    torch::Tensor last_action;

    // default values
    int action_refresh=0;
    int history_length = 10;
    //float init_pos[10] = {0.0,-0.07,0.628,-1.16,0.565,  0.0,0.07,0.628,-1.16,0.565};//important
    float init_pos[10] = {0.0,0.08,0.56,-1.12,-0.57,  0.0,-0.08,-0.56,1.12,0.57};//important
    float eu_ang_scale= 1;
    float omega_scale=  0.25;
    float pos_scale =   1.0;
    float vel_scale =   0.05;
    float lin_vel = 2.0;
    float ang_vel = 0.25;
    torch::jit::script::Module model;
    torch::DeviceType device;
private:

   
};

