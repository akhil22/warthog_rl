import pandas as pd
from env.WarthogEnv import WarthogEnv
import sys
import numpy as np
def get_pose_from_df(row):
    """ Computes pose data from data frame row

    Args:
        Data frame row
    Returns:
        Pose data 
    """
    x = row["x"]
    y = row["y"]
    th = row["th"]
    v = row["vel"]
    w = row["w"]
    return [x, y, th, v, w]
    
def get_action_from_df(row):
    """ computes the action command from data frame row 

    Args:
        data frame row
    returns:
        list of control commands
    """
    v_cmd = row["v_cmd"]
    w_cmd = row["w_cmd"]
    return [v_cmd, w_cmd]
def main():
    """ Reads the raw data from warthog environment from a file 
        and Computes observations (states) and action pair 
        to train a policy

        Args:
            sys.argv[1] : input file name (csv)
            sys.argv[1] : outpuf file name (csv)
    
    """
    df = pd.read_csv(sys.argv[1]) 
    out_file = sys.argv[2]
    out_file_h = open(out_file, "w")
    #out_file_h.writelines(f"obs,action\n")
    warthog_env = WarthogEnv(None)
    num_points = len(df.index)
#    num_points = 10
    num_traj_point = 10
    num_traj_point = num_points
    waypoint_dist = 1.5 
    print(num_points)
    #for each point in the file get the observation for training
    for i in range(0, num_points-1):
        print(i)
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
                    way_pose = get_pose_from_df(df.iloc[way_idx])
                    out_file_h.writelines(f"{way_pose[0]}, {way_pose[1]}, {way_pose[2]}, {way_pose[3]}, {way_pose[4]}\n")
                waypoint1 = np.array([df.iloc[way_idx]["x"],df.iloc[way_idx]["y"]])
                if np.linalg.norm(waypoint0 - waypoint1) >= waypoint_dist:
                    waypoint_list.append(get_pose_from_df(df.iloc[way_idx]))
                    way_pose = get_pose_from_df(df.iloc[way_idx])
                    out_file_h.writelines(f"{way_pose[0]}, {way_pose[1]}, {way_pose[2]}, {way_pose[3]}, {way_pose[4]}\n")
                    waypoint0 = np.array([df.iloc[way_idx]["x"],df.iloc[way_idx]["y"]])
            else:
                break
            k = k+1
        break
        out_file_h.close()
        return()

        #if there aren't sufficinet points in the list then populate 
        # the remaining list with zeros
        for j in range(k, num_traj_point - 1):
            waypoint_list.append(np.array([0., 0., 0., 0., 0.]))
        #print(waypoint_list)
        war_pose = get_pose_from_df(df.iloc[i])
        #warthog_env.set_pose(war_pose[0]+0.1, war_pose[1]+0.05, war_pose[2])
        warthog_env.set_pose(war_pose[0], war_pose[1], war_pose[2])
        warthog_env.set_twist(war_pose[3], war_pose[4])
        warthog_env.waypoints_list = []
        #append the observation in the waypoint list
        for i in waypoint_list:
            warthog_env.waypoints_list.append(np.array([i[0], i[1], i[2], i[3]]))
        warthog_env.num_waypoints = len(warthog_env.waypoints_list)
        obs = warthog_env.get_observation()
        #print(obs)
        #write the observation to the file
        for k in obs:
            out_file_h.writelines(f"{k}, ")
        out_file_h.writelines(f"{command[0]}, {command[1]}\n")
    out_file_h.close()




#       print(curr_row)

if __name__ == '__main__':
    main()
