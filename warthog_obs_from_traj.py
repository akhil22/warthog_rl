import pandas as pd
from env.WarthogEnv import WarthogEnv
import sys
import numpy as np
def get_pose_from_df(row):
    x = row["x"]
    y = row["y"]
    th = row["th"]
    v = row["vel"]
    w = row["w"]
    return [x, y, th, v, w]
    
def get_action_from_df(row):
    v_cmd = row["v_cmd"]
    w_cmd = row["w_cmd"]
    return [v_cmd, w_cmd]
def main():
    df = pd.read_csv(sys.argv[1]) 
    out_file = sys.argv[2]
    out_file_h = open(out_file, "w")
    out_file_h.writelines(f"obs,action\n")
    warthog_env = WarthogEnv(None)
    num_points = len(df.index)
    num_points = 10
    num_traj_point = 10
    print(num_points)
    for i in range(0, num_points-1):
        D = df.iloc[i]
        waypoint_list = []
        command=[0, 0]
        k = 1
        command = get_action_from_df(df.iloc[i])
        while len(waypoint_list) < num_traj_point: 
            way_idx = i+k
            if way_idx < len(df.index):
                if k == 1:
                    waypoint0 = np.array([df.iloc[way_idx]["x"],df.iloc[way_idx]["y"]])
                    waypoint_list.append(get_pose_from_df(df.iloc[way_idx]))
                waypoint1 = np.array([df.iloc[way_idx]["x"],df.iloc[way_idx]["y"]])
                if np.linalg.norm(waypoint0 - waypoint1) >=0.5:
                    waypoint_list.append(get_pose_from_df(df.iloc[way_idx]))
                    waypoint0 = np.array([df.iloc[way_idx]["x"],df.iloc[way_idx]["y"]])
            k = k+1
        for j in range(k, num_traj_point - 1):
            waypoint_list.append(np.array([0., 0., 0., 0., 0.]))
        print(waypoint_list)
        war_pose = get_pose_from_df(df.iloc[i])
        warthog_env.set_pose(war_pose[0], war_pose[1], war_pose[2])
        warthog_env.set_twist(war_pose[3], war_pose[4])
        warthog_env.waypoints_list = []
        for i in waypoint_list:
            warthog_env.waypoints_list.append(np.array([i[0], i[1], i[2], i[3]]))
        warthog_env.num_waypoints = len(warthog_env.waypoints_list)
        obs = warthog_env.get_observation()
        print(obs)
        out_file_h.writelines(f"{obs}, {command}\n")
    out_file_h.close()




#       print(curr_row)

if __name__ == '__main__':
    main()
