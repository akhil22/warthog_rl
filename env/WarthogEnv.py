from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import gym
import numpy as np
import math
from gym import spaces
import csv
import matplotlib as mpl
import time


class WarthogEnv(gym.Env):
    def __init__(self, waypoint_file):
        super(WarthogEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0.0, -1.5]),
                                       high=np.array([1.0, 1.5]),
                                       shape=(2, ))
        self.observation_space = spaces.Box(low=-100,
                                            high=1000,
                                            shape=(42, ),
                                            dtype=np.float)
        self.filename = waypoint_file
        plt.ion
        self.waypoints_list = []
        self.num_waypoints = 0
        self.pose = [0, 0, 0]
        self.twist = [0, 0]
        self.closest_idx = 0
        self.prev_closest_idx = 0
        self.closest_dist = math.inf
        self.num_waypoints = 0
        self.horizon = 10
        self.dt = 0.03
        self.ref_vel = []
        self._read_waypoint_file(self.filename)
        self.max_vel = 1
        self.fig = plt.figure(dpi=100, figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_box_aspect(1)
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.warthog_length = 0.5 / 2.0
        self.warthog_width = 1.0 / 2.0
        self.warthog_diag = math.sqrt(self.warthog_width**2 +
                                      self.warthog_length**2)
        self.plot_waypoints()
        self.rect = Rectangle((0., 0.),
                              self.warthog_width * 2,
                              self.warthog_length * 2,
                              fill=False)
        self.diag_ang = math.atan2(self.warthog_length, self.warthog_width)
        self.ax.add_artist(self.rect)
        self.prev_ang = 0
        self.n_traj = 100
        self.xpose = [0.] * 100
        self.ypose = [0.] * 100
        self.cur_pos, = self.ax.plot(self.xpose, self.ypose, '+g')
        self.t_start = self.ax.transData
        self.crosstrack_error = 0
        self.vel_error = 0
        self.phi_error = 0
        self.text = self.ax.text(1,
                                 2,
                                 f'vel_error={self.vel_error}',
                                 style='italic',
                                 bbox={
                                     'facecolor': 'red',
                                     'alpha': 0.5,
                                     'pad': 10
                                 },
                                 fontsize=12)
        #self.ax.add_artist(self.text)
        self.tprev = time.time()
        self.total_ep_reward = 0
        self.reward = 0

    def plot_waypoints(self):
        x = []
        y = []
        for i in range(0, self.num_waypoints):
            x.append(self.waypoints_list[i][0])
            y.append(self.waypoints_list[i][1])
        self.ax.plot(x, y, '+r')

    def sim_warthog(self, v, w):
        x = self.pose[0]
        y = self.pose[1]
        th = self.pose[2]
        v_ = self.twist[0]
        w_ = self.twist[1]
        dt = self.dt
        self.prev_ang = self.pose[2]
        self.pose[0] = x + v_ * math.cos(th) * dt
        self.pose[1] = y + v_ * math.sin(th) * dt
        self.pose[2] = th + w_ * dt
        self.twist[0] = v
        self.twist[1] = w

    def zero_to_2pi(self, theta):
        if theta < 0:
            theta = 2 * math.pi + theta
        elif theta > 2 * math.pi:
            theta = theta - 2 * math.pi
        return theta

    def pi_to_pi(self, theta):
        if theta < -math.pi:
            theta = theta + 2 * math.pi
        elif theta > math.pi:
            theta = theta - 2 * math.pi
        return theta

    def get_dist(self, waypoint, pose):
        xdiff = pose[0] - waypoint[0]
        ydiff = pose[1] - waypoint[1]
        return math.sqrt(xdiff * xdiff + ydiff * ydiff)

    def update_closest_idx(self, pose):
        idx = self.closest_idx
        self.closest_dist = math.inf
        for i in range(self.closest_idx, self.num_waypoints):
            dist = self.get_dist(self.waypoints_list[i], pose)
            if (dist <= self.closest_dist):
                self.closest_dist = dist
                idx = i
            else:
                break
        self.closest_idx = idx

    def get_theta(self, xdiff, ydiff):
        theta = math.atan2(ydiff, xdiff)
        return self.zero_to_2pi(theta)

    def get_observation(self):
        obs = [0] * (self.horizon * 4 + 2)
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
                yaw_error = self.pi_to_pi(self.waypoints_list[k][2] -
                                          vehicle_th)
                vel = self.waypoints_list[k][3]
                obs[j] = r
                obs[j + 1] = self.pi_to_pi(th - vehicle_th)
                obs[j + 2] = yaw_error
                obs[j + 3] = vel - twist[0]
            else:
                obs[j] = 0.
                obs[j + 1] = 0.
                obs[j + 2] = 0.
                obs[j + 3] = 0.
            j = j + 4
        obs[j] = twist[0]
        obs[j + 1] = twist[1]
        return obs

    def step(self, action):
        action[0] = np.clip(action[0], 0, 1) * 4.0
        action[1] = np.clip(action[1], -1, 1) * 2.5
        self.sim_warthog(action[0], action[1])
        self.prev_closest_idx = self.closest_idx
        obs = self.get_observation()
        done = False
        if self.closest_idx >= self.num_waypoints - 1:
            done = True
        #Calculating reward
        k = self.closest_idx
        xdiff = self.waypoints_list[k][0] - self.pose[0]
        ydiff = self.waypoints_list[k][1] - self.pose[1]
        th = self.get_theta(xdiff, ydiff)
        yaw_error = self.pi_to_pi(th - self.pose[2])
        self.phi_error = self.pi_to_pi(
            self.waypoints_list[self.closest_idx][2] - self.pose[2])
        self.vel_error = self.waypoints_list[k][3] - self.twist[0]
        self.crosstrack_error = self.closest_dist * math.sin(yaw_error)
        if (math.fabs(self.crosstrack_error) > 1.5
                or math.fabs(self.phi_error) > 1.4):
            done = True
        self.reward = (2.0 - math.fabs(self.crosstrack_error)) * (
            4.0 - math.fabs(self.vel_error)) * (math.pi / 3. -
                                                math.fabs(self.phi_error))
        if (self.prev_closest_idx == self.closest_idx
                or math.fabs(self.vel_error) > 1.5):
            self.reward = 0
        self.total_ep_reward = self.total_ep_reward + self.reward
        #self.render()
        return obs, self.reward, done, {}

    def reset(self):
        self.total_ep_reward = 0
        if (self.max_vel >= 5):
            self.max_vel = 1
        idx = np.random.randint(self.num_waypoints, size=1)
        idx = idx[0]
        self.closest_idx = 0
        self.prev_closest_idx = 0
        self.pose[0] = self.waypoints_list[idx][0] + 0.1
        self.pose[1] = self.waypoints_list[idx][1] + 0.1
        self.pose[2] = self.waypoints_list[idx][2] + 0.01
        self.xpose = [self.pose[0]] * self.n_traj
        self.ypose = [self.pose[1]] * self.n_traj
        self.twist = [0., 0., 0.]
        for i in range(0, self.num_waypoints):
            if (self.ref_vel[i] > self.max_vel):
                self.waypoints_list[i][3] = self.max_vel
            else:
                self.waypoints_list[i][3] = self.ref_vel[i]
        self.max_vel = self.max_vel + 1
        obs = self.get_observation()
        return obs

    def render(self, mode='human'):
        self.ax.set_xlim([self.pose[0] - 4, self.pose[0] + 4])
        self.ax.set_ylim([self.pose[1] - 4, self.pose[1] + 4])
        total_diag_ang = self.diag_ang + self.pose[2]
        xl = self.pose[0] - self.warthog_diag * math.cos(total_diag_ang)
        yl = self.pose[1] - self.warthog_diag * math.sin(total_diag_ang)
        #self.rect.set_xy((0, 0))
        #t = mpl.transforms.Affine2D().rotate_around(xl, yl, self.pose[2])
        #t = mpl.transforms.Affine2D().rotate(self.pose[2])
        #self.rect._angle = self.pose[2]
        #self.rect.set_xy((xl, yl))
        #t = rect.get_patch_transform()
        #self.rect.set_transform(t + self.t_start)
        #self.rect.set_transform(t)
        #self.rect.set_width(self.warthog_width * 2)
        #self.rect.set_height(self.warthog_length * 2)
        #del self.rect
        self.rect.remove()
        self.rect = Rectangle((xl, yl), self.warthog_width * 2,
                              self.warthog_length * 2,
                              180.0 * self.pose[2] / math.pi)
        self.text.remove()
        self.text = self.ax.text(
            self.pose[0] + 1,
            self.pose[1] + 2,
            f'vel_error={self.vel_error:.3f}\nclosest_idx={self.closest_idx}\ncrosstrack_error={self.crosstrack_error:.3f}\nReward={self.reward:.4f}\nwarthog_vel={self.twist[0]:.3f}\nphi_error={self.phi_error*180/math.pi:.4f}\nsim step={time.time() - self.tprev:.4f}\nep_reward={self.total_ep_reward:.4f}\nmax_vel={self.max_vel:.4f}',
            style='italic',
            bbox={
                'facecolor': 'red',
                'alpha': 0.5,
                'pad': 10
            },
            fontsize=10)
        #print(time.time() - self.tprev)
        self.tprev = time.time()
        #self.ax.add_artist(self.text)
        self.ax.add_artist(self.rect)
        self.xpose.append(self.pose[0])
        self.ypose.append(self.pose[1])
        del self.xpose[0]
        del self.ypose[0]
        self.cur_pos.set_xdata(self.xpose)
        self.cur_pos.set_ydata(self.ypose)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _read_waypoint_file(self, filename):
        with open(filename) as csv_file:
            pos = csv.reader(csv_file, delimiter=',')
            for row in pos:
                #utm_cord = utm.from_latlon(float(row[0]), float(row[1]))
                utm_cord = [float(row[0]), float(row[1])]
                #phi = math.pi/4
                phi = 0.
                xcoord = utm_cord[0] * math.cos(phi) + utm_cord[1] * math.sin(
                    phi)
                ycoord = -utm_cord[0] * math.sin(phi) + utm_cord[1] * math.cos(
                    phi)
                #   self.waypoints_list.append(np.array([xcoord, ycoord, float(row[2]),float(row[3])]))
                #self.waypoints_list.append(np.array([xcoord, ycoord, float(row[2]),2.5]))
                self.waypoints_list.append(
                    np.array([
                        utm_cord[0], utm_cord[1],
                        float(row[2]),
                        float(row[3])
                    ]))
                self.ref_vel.append(float(row[3]))
            # self.waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]), 1.5]))
            for i in range(0, len(self.waypoints_list) - 1):
                xdiff = self.waypoints_list[i +
                                            1][0] - self.waypoints_list[i][0]
                ydiff = self.waypoints_list[i +
                                            1][1] - self.waypoints_list[i][1]
                self.waypoints_list[i][2] = self.zero_to_2pi(
                    self.get_theta(xdiff, ydiff))
            self.waypoints_list[i + 1][2] = self.waypoints_list[i][2]
            self.num_waypoints = i + 2
        pass
