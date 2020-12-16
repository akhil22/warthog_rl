from matplotlib import pyplot as plt
import gym
import numpy as np
import math
from gym import spaces
import csv
class WarthogEnv(gym.Env):
    def __init__(self, waypoint_file):
        super(WarthogEnv, self).__init__()
        self.action_space = spaces.Box(low = np.array([0.0,-1.]), high = np.array([0.,1.]), shape=(2,))
        self.observation_space = spaces.Box(low = -100, high = 1000, shape=(42,),dtype = np.float)
        self.filename = waypoint_file
        self.waypoints_list = []
        self.num_waypoints = 0
        self.pose = [0, 0, 0]
        self.twist = [0, 0]
        self.closest_idx = 0
        self.closest_dist = math.inf
        self.num_waypoints = 0
        self.horizon = 10
        self.dt = 0.03
        self._read_waypoint_file(self.filename)
    def sim_warthog(self, v, w):
        x = self.pose[0]
        y = self.pose[1]
        th = self.pose[2]
        v_ = self.twist[0]
        w_ = self.twist[1]
        dt = self.dt
        self.pose[0] = x + v_*math.cos(th)*dt
        self.pose[1] = y + v_*math.sin(th)*dt
        self.pose[2] = th + w*dt
        self.twist[0] = v
        self.twist[1]= w
    def zero_to_2pi(self, theta):
        if theta < 0:
            theta = 2*math.pi + theta
        elif theta > 2*math.pi:
            theta = theta - 2*math.pi
        return theta
    def pi_to_pi(self, theta):
        if theta < -math.pi:
            theta = theta + 2*math.pi
        elif theta > math.pi:
            theta = theta - 2*math.pi
        return theta
    def get_dist(self, waypoint, pose):
        xdiff = pose[0] - waypoint[0]
        ydiff = pose[1] - waypoint[1]
        return math.sqrt(xdiff*xdiff + ydiff*ydiff)
    def update_closest_idx(self, pose):
        idx = self.closest_idx
        self.closest_dist = math.inf
        for i in range(self.closest_idx, self.num_waypoints):
            dist = self.get_dist(self.waypoints_list[i], pose)
            if(dist <= self.closest_dist):
                self.closest_dist = dist
                idx = i
            else:
                break
        self.closest_idx = idx
    def get_theta(self, xdiff, ydiff):
        theta = math.atan2(ydiff, xdiff)
        return self.zero_to_2pi(theta)
    def get_observation(self):
        obs = [0]*(self.horizon*4 + 2)
        pose = self.pose
        twist = self.twist
        self.update_closest_idx(pose)
        j = 0
        for i in range(0, self.horizon):
            k = i + self.closest_idx
            if k < self.num_waypoints:
                r = self.get_dist(self.waypoints_list[k], pose)
                xdiff = self.waypoints_list[k][0] - pose[0]
                ydiff = self.waypoints_list[k][1] - pose[1]
                th = self.get_theta(xdiff, ydiff)
                vehicle_th = self.zero_to_2pi(pose[2])
                #vehicle_th = -vehicle_th
                #vehicle_th = 2*math.pi - vehicle_th
                yaw_error = self.pi_to_pi(self.waypoints_list[k][2] - vehicle_th)
                vel = self.waypoints_list[k][3]
                obs[j] = r
                obs[j+1] = self.pi_to_pi(th - vehicle_th)
                obs[j+2] = yaw_error
                obs[j+3] = vel - twist[0]
            else:
                obs[j] = 0.
                obs[j+1] = 0.
                obs[j+2] = 0.
                obs[j+3] = 0.
            j = j+4
        obs[j] = twist[0]
        obs[j+1] = twist[1]
        return obs
    def step(self, action):
        self.sim_warthog(action[0], action[1])
        obs = self.get_observation()
        done = False
        if self.closest_idx >= self.num_waypoints - 1:
            done = True
        return obs, 0, done, {}
    def reset(self):
        idx = np.random.randint(self.num_waypoints, size=1)
        idx = idx[0]
        self.pose[0] = self.waypoints_list[idx][0] + 0.1
        self.pose[1]= self.waypoints_list[idx][1] + 0.1
        self.pose[2] = self.waypoints_list[idx][2] + 0.01
        self.twist = [0., 0., 0.]
        print(self.pose)
    def render(self, mode='huaman'):
        pass
    def _read_waypoint_file(self, filename):
        with open(filename) as csv_file:
            pos = csv.reader(csv_file, delimiter=',')
            for row in pos:
                #utm_cord = utm.from_latlon(float(row[0]), float(row[1]))
                utm_cord = [float(row[0]), float(row[1])]
                #phi = math.pi/4
                phi = 0.
                xcoord = utm_cord[0]*math.cos(phi) + utm_cord[1]*math.sin(phi)
                ycoord = -utm_cord[0]*math.sin(phi) + utm_cord[1]*math.cos(phi)
             #   self.waypoints_list.append(np.array([xcoord, ycoord, float(row[2]),float(row[3])]))
                #self.waypoints_list.append(np.array([xcoord, ycoord, float(row[2]),2.5]))
                self.waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]),float(row[3])]))
               # self.waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]), 1.5]))
            for i in range(0, len(self.waypoints_list) - 1):
                xdiff = self.waypoints_list[i+1][0] - self.waypoints_list[i][0]
                ydiff = self.waypoints_list[i+1][1] - self.waypoints_list[i][1]
                self.waypoints_list[i][2] = self.zero_to_2pi(self.get_theta(xdiff, ydiff))
            self.waypoints_list[i+1][2] = self.waypoints_list[i][2]
            self.num_waypoints = i+2
        pass
