# Data Processing

## Real Robot

1. Collect data in a bag file running the warthog with following topics: 
   * "/warthog_velocity_controller/cmd_vel" (Command from remote), "/odometry/filtered" (Local localiztaion), "/odometry/fitered2" (GPS), "/odometry/filtered_map" (Global localization)"

2. Start the bag file and run:
   ```python
    python warthog_real_data_collector.py
    ```
    * ROS Parameters:
      * cmd_vel_topic: command velocity topic to get commanded v and w
      * odom_topic: odom topic to get warthog's v and w 
      * gps_odom_topic: odom topic to get poses (x, y, th)
      * out_file: name of the output file to save pose and command data
3. Previous command will save the poses and command in the default output file: real_remote_pose_ext_war_gps.csv.
Run the following command to compute the (cmd,observations) tuples (for training RL policy) from this file and save them real_remote_obs.csv
   ```python
   python obs_from_traj.py real_remote_pose_ext_war_gps.csv real_remote_obs.csv
   ```
4. To train a saved RL policy (./policy/rl_policy.zip) using the above observations run the following command:
   ```python
   python policy_train.py ./policy/rl_policy real_remote_obs.csv ./policy/rl_policy_after_train
   ```
   this will save the trained policy in ./policy/rl_policy_after_train.zip


