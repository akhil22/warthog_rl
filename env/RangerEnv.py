from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import gym
import numpy as np
import math
from gym import spaces
import csv
import matplotlib as mpl
import time
import sys
import mavs_interface as mavs
import mavs_python_paths


class RangerEnv(gym.Env):
    def __init__(self, waypoint_file):
        super(RangerEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.]),
                                       high=np.array([1.0, 1.0, 1.]),
                                       shape=(3, ))
        self.observation_space = spaces.Box(low=-100,
                                            high=1000,
                                            shape=(42, ),
                                            dtype=np.float)
        self.filename = waypoint_file
        plt.ion
        self.mavs_data_path = mavs_python_paths.mavs_data_path
        self.waypoints_list = []
        self.num_waypoints = 0
        self.pose = [0, 0, 0]
        self.twist = [0, 0]
        self.closest_idx = 0
        self.prev_closest_idx = 0
        self.closest_dist = math.inf
        self.num_waypoints = 0
        self.horizon = 10
        self.dt = 0.06
        self.ref_vel = []
        self.axis_size = 20
        if self.filename is not None:
            self._read_waypoint_file(self.filename)
        self.veh = mavs.MavsRp3d()
        self.mavs_env = mavs.MavsEnvironment()
        self._mav_scene_init()
        self.max_vel = 1
        self.fig = plt.figure(dpi=100, figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_box_aspect(1)
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.warthog_length = 0.5 / 2.0
        self.warthog_width = 1.0 / 2.0
        self.warthog_diag = math.sqrt(self.warthog_width**2 +
                                      self.warthog_length**2)
        if self.filename is not None:
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
        self.ep_steps = 0
        self.max_ep_steps = 700
        self.tprev = time.time()
        self.total_ep_reward = 0
        self.reward = 0
        self.action = [0., 0.]
        self.prev_action = [0., 0.]
        self.omega_reward = 0
        self.vel_reward = 0
        self.is_delayed_dynamics = False
        self.delay_steps = 5
        self.v_delay_data = [0.] * self.delay_steps
        self.w_delay_data = [0.] * self.delay_steps

    def set_pose(self, x, y, th):
        self.veh.SetInitialPosition(x, y, 0.)
        self.veh.SetInitialHeading(th)

    def set_twist(self, v, w):
        self.tiwst = [v, w]

    def plot_waypoints(self):
        x = []
        y = []
        for i in range(0, self.num_waypoints):
            x.append(self.waypoints_list[i][0])
            y.append(self.waypoints_list[i][1])
        self.ax.plot(x, y, '+r')

    def sim_ranger(self, accel, brake, steer):
        self.veh.Update(self.mavs_env, accel, steer, brake, self.dt)
        x = self.pose[0]
        y = self.pose[1]
        th = self.pose[2]
        v_ = self.twist[0]
        w_ = self.twist[1]
        self.twist[0] = v
        self.twist[1] = w
        if self.is_delayed_dynamics:
            v_ = self.v_delay_data[0]
            w_ = self.w_delay_data[0]
            del self.v_delay_data[0]
            del self.w_delay_data[0]
            self.v_delay_data.append(v)
            self.w_delay_data.append(w)
            self.twist[0] = self.v_delay_data[0]
            self.twist[1] = self.v_delay_data[1]
        dt = self.dt
        self.prev_ang = self.pose[2]
        self.pose[0] = x + v_ * math.cos(th) * dt
        self.pose[1] = y + v_ * math.sin(th) * dt
        self.pose[2] = th + w_ * dt

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
        self.ep_steps = self.ep_steps + 1
        action[0] = np.clip(action[0], 0, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[1], -1, 1)
        self.action = action
        self.sim_ranger(action[0], action[1], action[2])
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
        if self.ep_steps == self.max_ep_steps:
            done = True
            self.ep_steps = 0
        self.reward = (2.0 - math.fabs(self.crosstrack_error)) * (
            4.5 - math.fabs(self.vel_error)) * (
                math.pi / 3. - math.fabs(self.phi_error)) - math.fabs(
                    self.action[0] -
                    self.prev_action[0]) - 2 * math.fabs(self.action[1])
        self.omega_reward = -2 * math.fabs(self.action[1])
        self.vel_reward = -math.fabs(self.action[0] - self.prev_action[0])
        #self.reward = (2.0 - math.fabs(self.crosstrack_error)) * (
        #    4.0 - math.fabs(self.vel_error)) * (math.pi / 3. -
        #                                        math.fabs(self.phi_error)) - math.fabs(self.action[0] - self.prev_action[0]) - 1.3*math.fabs(self.action[1] - self.prev_action[1])
        self.prev_action = self.action
        #if (self.prev_closest_idx == self.closest_idx
        #        or math.fabs(self.vel_error) > 1.5):
        if self.waypoints_list[k][3] >= 2.5 and math.fabs(
                self.vel_error) > 1.5:
            self.reward = 0
        elif self.waypoints_list[k][3] < 2.5 and math.fabs(
                self.vel_error) > 0.5:
            self.reward = 0
        self.total_ep_reward = self.total_ep_reward + self.reward
        #self.render()
        return obs, self.reward, done, {}

    def reset(self):
        self.total_ep_reward = 0
        if (self.max_vel >= 5):
            self.max_vel = 1
        idx = np.random.randint(self.num_waypoints, size=1)
        #idx = [0]
        idx = idx[0]
        #idx = 880
        self.closest_idx = idx
        self.prev_closest_idx = idx
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
        #self.max_vel = 2
        self.max_vel = self.max_vel + 1
        obs = self.get_observation()
        return obs

    def render(self, mode='human'):
        self.ax.set_xlim([
            self.pose[0] - self.axis_size / 2.0,
            self.pose[0] + self.axis_size / 2.0
        ])
        self.ax.set_ylim([
            self.pose[1] - self.axis_size / 2.0,
            self.pose[1] + self.axis_size / 2.0
        ])
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
        self.rect = Rectangle((xl, yl),
                              self.warthog_width * 2,
                              self.warthog_length * 2,
                              180.0 * self.pose[2] / math.pi,
                              facecolor='blue')
        self.text.remove()
        self.text = self.ax.text(
            self.pose[0] + 1,
            self.pose[1] + 2,
            f'vel_error={self.vel_error:.3f}\nclosest_idx={self.closest_idx}\ncrosstrack_error={self.crosstrack_error:.3f}\nReward={self.reward:.4f}\nwarthog_vel={self.twist[0]:.3f}\nphi_error={self.phi_error*180/math.pi:.4f}\nsim step={time.time() - self.tprev:.4f}\nep_reward={self.total_ep_reward:.4f}\nmax_vel={self.max_vel:.4f}\nomega_reward={self.omega_reward:.4f}\nvel_reward={self.vel_error:.4f}',
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

    def _mav_scene_init(self):
        random_scene = mavs.MavsRandomScene()
        random_scene.terrain_width = 250.0
        random_scene.terrain_length = 250.0
        random_scene.lo_mag = 0.0
        random_scene.hi_mag = 0.05
        random_scene.mesh_resolution = 0.3
        random_scene.plant_density = 0.0
        random_scene.trail_width = 0.0
        random_scene.track_width = 0.0
        random_scene.wheelbase = 0.0
        random_scene.surface_roughness_type = "variable"
        scene_name = 'bumpy_surface'
        random_scene.basename = scene_name
        random_scene.eco_file = 'american_pine_forest.json'
        #random_scene.eco_file = 'american_southwest_desert.json'
        random_scene.path_type = 'Ridges'
        random_scene.CreateScene()

        # Create a MAVS environment and add the scene to it
        #env.SetScene(scene.scene)
        self.mavs_env.SetScene(random_scene.scene)

        # Set environment properties
        self.mavs_env.SetTime(13)  # 0-23
        self.mavs_env.SetFog(0.0)  # 0.0-100.0
        self.mavs_env.SetSnow(0.0)  # 0-25
        self.mavs_env.SetTurbidity(7.0)  # 2-10
        self.mavs_env.SetAlbedo(0.1)  # 0-1
        self.mavs_env.SetCloudCover(0.5)  # 0-1
        self.mavs_env.SetRainRate(0.0)  # 0-25
        self.mavs_env.SetWind([2.5, 1.0])  # Horizontal windspeed in m/s

        #Create and load a MAVS vehicle
        # vehicle files are in the mavs "data/vehicles/rp3d_vehicles" folder
        #veh_file = 'forester_2017_rp3d.json'
        veh_file = 'mrzr4_tires.json'
        #veh_file = 'clearpath_warthog.json'
        #veh_file = 'hmmwv_rp3d.json'
        #veh_file = 'mrzr4.json'
        #veh_file = 'sedan_rp3d.json'
        #veh_file = 'cucv_laredo_rp3d.json'
        self.veh.Load(mavs_data_path + '/vehicles/rp3d_vehicles/' + veh_file)
        # Starting point for the vehicle
        #veh.SetInitialPosition(-52.5, 7.5, 0.0) # in global ENU
        self.veh.SetInitialPosition(100.0, 0.0, 0.0)  # in global ENU
        #veh.SetInitialPosition(65.125, 35.0, 0.0) # in global ENU
        # Initial Heading for the vehicle, 0=X, pi/2=Y, pi=-X
        self.veh.SetInitialHeading(0.0)  # in radians
        #veh.SetInitialHeading(-1.57) # in radians
        self.veh.Update(self.mavs_env, 0.0, 0.0, 1.0, 0.000001)

        # Create a window for driving the vehicle with the W-A-S-D keys
        # window must be highlighted to input driving commands
        drive_cam = mavs.MavsCamera()
        # nx,ny,dx,dy,focal_len
        drive_cam.Initialize(256, 256, 0.0035, 0.0035, 0.0035)
        # offset of camera from vehicle CG
        drive_cam.SetOffset([-10.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        # Set camera compression and gain
        #drive_cam.SetGammaAndGain(0.6,1.0)
        drive_cam.SetGammaAndGain(0.5, 2.0)
        # Turn off shadows for this camera for efficiency purposes
        drive_cam.RenderShadows(True)

        front_cam = mavs.MavsCamera()
        # nx,ny,dx,dy,focal_len
        front_cam.Initialize(256, 256, 0.0035, 0.0035, 0.0035)
        # offset of camera from vehicle CG
        angle = 135.0
        front_cam.SetOffset([3.5, -2.6, 0.0], [
            math.cos(0.5 * math.radians(angle)), 0.0, 0.0,
            math.sin(0.5 * math.radians(angle))
        ])
        #angle = 90.0
        #front_cam.SetOffset([1.5,-2.6,0.0],[math.cos(0.5*math.radians(angle)),0.0, 0.0, math.sin(0.5*math.radians(angle))])
        # Set camera compression and gain
        #drive_cam.SetGammaAndGain(0.6,1.0)
        front_cam.SetGammaAndGain(0.5, 2.0)
        # Turn off shadows for this camera for efficiency purposes
        front_cam.RenderShadows(True)
