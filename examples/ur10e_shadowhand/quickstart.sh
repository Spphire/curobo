gnome-terminal -t "ros_loop" -x bash -c "sshpass -p shadow251nuc ssh user@10.9.11.2;"

gnome-terminal -t "ros_gui" -x bash -c "sshpass -p shadow251nuc ssh user@10.9.11.2;"
sleep 0.5s
# gnome-terminal -t "ros_side" -x bash -c "source ~/workspace/shadow_ros/devel/setup.bash;cd ~/workspace/syb/curobo;exec bash;"

# gnome-terminal -t "fast_api" -x bash -c "cd ~/workspace/syb/curobo;conda activate rfplanner;"