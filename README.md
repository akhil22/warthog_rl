# Data Processing

## Real Robot

1. Collect data in a bag file running the warthog with following topics: 
  * "/warthog_velocity_controller/cmd_vel" (Command from remote), "/odometry/filtered" (Local localiztaion), "/odometry/fitered2" (GPS), "/odometry/filtered_map" (Global localization)"

2. Start the bag file and run:
  * ```python
    python warthog_real_data_collector.py
    ```
