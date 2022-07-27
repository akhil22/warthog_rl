"""get observation from the trajectory csv file and save it on disk.

This file reads trajectory data from trajectory file, computes observation 
from the trajectory and saves it to the output file. The file can be collected from various different sources. Every line in the csv file looks like the following:
    x, y, th,v, vel, w, ep_start 
x: x position 
y: y position 
th: yaw
v: current velocity
vel: commanded velocity
w: commanded angular velocity
ep_start: if episode started from this step 

every nth line in the trajectory represents vehicle state at nth time step this file is primarily used to plot waypoints from trajectory collected during PPO learning to see if the trajectory is good enough to be used for supervised learning.

    Typical usage example:

    python traj_vis traj_filename.csv output_filename.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from tqdm import tqdm
from env.WarthogEnv import WarthogEnv

def get_waypoint_from_df(row):
    """ Computes pose data from data frame row

    Args:
        Data frame row
    Returns:
        Pose data 
    """
    x = row["x"]
    y = row["y"]
    th = row["th"]
    v = row["v_"]
    w = row["w_"]
    v_cmd = row["v"]
    w_cmd = row["w"]
    return np.array([x, y, th, v, w, v_cmd, w_cmd])
def is_ep_end(row):
    return row["ep_start"]
def main():
    """ Read the csv file and plot the waypoints.

    Args:
        sys.argv[1]: input csv file name
        sys.argv[2]: output csv file name
    """
    df = pd.read_csv(sys.argv[1])
    out_file_h = open(sys.argv[2], "w")
    print(sys.argv[1])
    num_steps = len(df.index)
    episode_traj = []
    episode_index_info = []
    waypoint_dist = 0.5
    #start = 500000;
    start = 0;
    previ = start
    #total_st = 500000
    total_st = num_steps-1
    pbar = tqdm(total=total_st)
    i = start
    print("getting episodes")
    while i < start+total_st:
        ep_end = is_ep_end(df.iloc[i])
        if not ep_end:
            i = i+1
            continue
        ep_end = False
        waypoint_list=[] 
        episode_index = []
        episode_index.append(i)
        prev_waypoint = get_waypoint_from_df(df.iloc[i])
        waypoint_list.append(prev_waypoint)
        ep_dist = 0
        k = i+1
        while not ep_end and k < num_steps:
            curr_waypoint = get_waypoint_from_df(df.iloc[k])
            dist_from_prev_wp = np.linalg.norm(curr_waypoint[0:2] - prev_waypoint[0:2])
            ep_end = is_ep_end(df.iloc[k])
            if ep_end:
                episode_traj.append(waypoint_list)
                i = k
                break
            if dist_from_prev_wp >= waypoint_dist:
                waypoint_list.append(curr_waypoint)
                prev_waypoint = curr_waypoint
                ep_dist = ep_dist + dist_from_prev_wp
            k = k+1
        episode_index.append(i-1)
        episode_index_info.append(episode_index)
        pbar.update(i-previ)
        previ = i
    save_way_points = True
    if save_way_points:
        way_file = open("temp_way.csv", 'w')
        for way in episode_traj[0]:
            way_file.writelines(f"{way[0]},{way[1]},{way[2]},{way[3]},{way[4]}\n")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    warthog_env = WarthogEnv(None,None)
    print("getting observation")
    #plot_way_and_obs = False
    plot_way_and_obs = True
    plt.ion()
    for j in tqdm(range(0,len(episode_traj))): 
        #print(episode_traj[j])
        #print(episode_index_info[j])
        warthog_env = WarthogEnv(None,None)
        plt.close('all')
        waypoint_list = episode_traj[j]
        if(len(waypoint_list) < 25):
            continue
        start_id = episode_index_info[j][0]
        end_id = episode_index_info[j][1]
        warthog_env.waypoints_list = []
        for i in waypoint_list:
            warthog_env.waypoints_list.append([i[0], i[1], i[2], i[3]])
        warthog_env.num_waypoints = len(warthog_env.waypoints_list)
        for k in range(start_id, end_id):
            war_pose = get_waypoint_from_df(df.iloc[k])
            warthog_env.set_pose(war_pose[0], war_pose[1], war_pose[2])
            warthog_env.set_twist(war_pose[3], war_pose[4])
            obs = warthog_env.get_observation()
            command = [war_pose[5], war_pose[6]]
            if plot_way_and_obs and j>=1:
                ob_way = []
                for j in range(0,10):
                    x_ob = war_pose[0] + obs[j*4]*np.cos(obs[j*4+1]+war_pose[2])
                    y_ob = war_pose[1] + obs[j*4]*np.sin(obs[j*4+1]+war_pose[2])
                    ob_way.append([x_ob, y_ob])
                plt.plot([x[0] for x in waypoint_list], [x[1] for x in waypoint_list])
                plt.plot([x[0] for x in ob_way], [x[1] for x in ob_way], '+r') 
                plt.plot(war_pose[0], war_pose[1], '*g')
                plt.draw()
                plt.pause(0.001)
                plt.clf()
            for ob in obs:
                out_file_h.writelines(f"{ob}, ")
            out_file_h.writelines(f"{command[0]}, {command[1]}\n")
if __name__ =="__main__":
    main()


















