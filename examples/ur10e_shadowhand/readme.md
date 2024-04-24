
# Useful file
ros_side.py

fastapi_receiver.py

ur10e_controller.py

rfplanner.py

# Usage
Release machine and start nuc

connect ssh to the nuc
```bash
ssh user@10.9.11.2
```
password: shadow251nuc

In ssh terminal:
```bash
roscd sr_robot_launch/scripts/
./shadow_right_arm_hand_hardware_control_loop.sh 
```

After fully run command above(important for rviz), in another ssh terminal:
```bash
roscd sr_robot_launch/scripts/
./shadow_right_arm_hand_gui.sh 
```

open a normal terminal:
```bash
source ~/workspace/shadow_ros/devel/setup.bash #to activate ros environment
cd ~/workspace/syb/curobo
python3 examples/ur10e_shadowhand/ros_side.py
```

open another normal terminal:
```bash
conda activate rfplanner #to activate rfplanner+curobo environment
cd ~/workspace/syb/curobo
python examples/ur10e_shadowhand/fastapi_receiver_thread.py
```

In Quest3, make sure wifi connected to 'ASUS_80_5G', and run unity app

hold menu button (三) to set blue arrow facing to east

press x button to start

press y button to stop

