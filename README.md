![1756024175889](doc/H1.png)

# Isaac Gym Environments for Legged Robots

This repository provides the environment used to train Alpha-Human-H1(and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.

It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and ran dom pushes during training.

**Maintainer**: Darren Wang

**Contact**: wang.gov@163.com; wang.gov@outlook.com

# Installation

## Preparation before installation

Before installation, you need to install the isaacgym and legged_gym related environment.

```
https://github.com/leggedrobotics/legged_gym
```

After installation, it includes various required environments.

## installation

```
conda create -n Alpha_Human_gym python=3.8
conda activate Alpha_Human_gym
pip install numpy==1.23.5 mujoco==2.3.7 mujoco-py mujoco-python-viewer 
pip install dm_control==1.0.14 opencv-python matplotlib einops tqdm packaging h5py ipython getkey wandb chardet h5py_cache tensorboard pyquaternion pyyaml rospkg pexpect
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge cmake=3.25
```

> cd /isaacgym/python/ && pip install -e .

# RL training

## Train

> ```
> conda activate Alpha_Human_gym
> cd Alpha_Human_gym/scripts/
> python train.py
> ```

```
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/user/anaconda3/envs/Alpha_Human_gym/lib/
```

```
AttributeError: module 'numpy' has no attribute 'float'.
pip uninstall numpy==1.20
pip install numpy==1.23.5
```

## Play

After completing the walking  training, the system will generate a models in the log folder, and the corresponding addresses need to be modified in `PLAY_DIR`.  The address of the `xxx.pt` file here needs to be determined based on the storage location of your training results.

```
PLAY_DIR ='logs/h1_constraint_trot/May31_21-37-30_/model_30000.pt'
```

then:

```
python play.py
```

# Sim-to-sim

```
python sim2sim.py
```

# Sim-to-real

Alpha-Human-H1 have two methods for Sim to real: one is to send control signals to Alpha-Human-H1 through UDP on the user's computer, and the other is to send the model to the computer on the Alpha-Human-H1,  then computer will automatically control the Alpha-Human-H1.

## Find IP address

Connect to Alpha-Human-H1's wifi, which is usually named `H1-2.4G`. Then enter the wifi IP in the browser , to log in to the wifi.

The default WiFi address for Alpha-Human-H1 is`192.168.1.1`, the account is `admin`, and the password is `admin`.

Then,  in the browser, search for the IP addresses of `focal-server` and `nvidia-desktop`.

For example,  the IP address of `focal-server`(odroid) is `192.168.1.242`,  and the IP address of`nvidia-desktop` (nvidia nano) is `192.168.1.129`.

## Nvidia Nano control

By default, Alpha-Human-H1 is controlled using RL on Nvidia Nano,  so unless there is a special need,  you can always control Alpha-Human-H1 on Nvidia Nano.

```
# Copy the walking model you trained into Nvidia Nano.
scp ./model/*jitt.pt nvidia@192.168.1.129:/home/nvidia/sim2real/model/ 
# SSH login to Nvidia Nano
ssh nvidia@192.168.1.129  # password：1
conda activate /home/nvidia/rlgpu
cd sim2real/src/
# use vim to change the IP address to the Alpha-Human-H1 control board (odroid) IP address.
sudo vim sim2real_h1.cpp 
# then compile sim2real project
cd sim2real/build/
cmake ..
make -j4
./sim2real_h1
```

If the communication with Alpha-Human-H1 is successfully established, the terminal will print the atction data.

```
act send:-0.76 2.1 5.7 0.3 3.2 -0.19 -2.1 1.6 -6.3 -1.2
```

Then follow the instructions to operate Alpha-Human-H1, which can be found in the `doc` folder.

## User computer control

Before compiling sim2real, it is necessary to download libtorch from the PyTorch official website. We recommend version `2.4.0+cu121`, and users can also try other versions.

Then unzip the downloaded compressed file to the sim2real folder, the download address is as follows:

```
https://pytorch.org/
```

Open the` /H1/sim2real/src/sim2real_h1.cpp` file, change the IP address to the control board (odroid) IP address.

At this point, you need to unplug the Nvidia Nano port on the router , then connect your computer to this port using a network cable.  then:

```
# string UDP_IP="192.168.XX.XX"; you should update your robot(odroid) IP address
cd /H1/sim2real/build
cmake ..
make -j4
```

Then execute the executable file to establish communication with Alpha-Human-H1:

```
sudo ./sim2real_h1
```

If the communication with Alpha-Human-H1 is successfully established, the terminal will print the atction data.

```
act send:-0.76 2.1 5.7 0.3 3.2 -0.19 -2.1 1.6 -6.3 -1.2
```

Then follow the instructions to operate Alpha-Human-H1, which can be found in the `doc` folder.

# Hardware firmware update

The robot firmware includes three parts: stm32, odroid, and nvidia nano. The firmware is all placed in the `hardware` folder, and the update method can be found in the `doc` folder.

# Acknowledgement

* [Tinker](https://github.com/golaced/OmniBotSeries-Tinker): a high-performance bipedal robot project that combines Pixar and Disney BDX.
* [Humanoid-Gym](https://github.com/roboterax/humanoid-gym):a phase based reinforcement learning training environment with Sim2Sim support.
* [legged_gym](https://github.com/leggedrobotics/legged_gym): The earliest open-source robot training environment project.
