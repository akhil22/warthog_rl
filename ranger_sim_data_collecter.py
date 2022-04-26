from env.RangerEnv_vis import RangerEnv
from threading import Thread, Lock
import rospy
import threading
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from matplotlib import pyplot as plt

accel = 0
brake = 0
steer = 0
mutex = Lock()


class DataCollector:
    """ Simulation data collector

    Use the cmd_vel_topic to drive a warthog in simulation and collect 
    dynamics data 

    Attributes:
        env: Warthog gym environment
        cmd_vel_topic: ros topic to get command velocity
        out_file: name of the output file in which to save observation
        file_h : output file handler
    """
    def __init__(self):
        """Initialize Data collector with gyn environment
        command velocity topic and output file name
        """
        self.env = RangerEnv(None)
        self.cmd_vel_topic = rospy.get_param(
            '~cmd_vel_topic', 'warthog_velocity_controller/cmd_vel')
        #self.joy_topic = rospy.get_param('~joy_topic', 'joy_orig')
        self.joy_topic = rospy.get_param('~joy_topic', 'game_control/joy')
        self.out_file = rospy.get_param('~out_file_name',
                                        'ranger_remote_sim_poses3.csv')
        self.file_h = open(self.out_file, 'w')
        self.file_h.writelines(f"x,y,th,vel,w,v_cmd,w_cmd\n")
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_cb)
        rospy.Subscriber(self.joy_topic, Joy, self.joy_cb)

    def render_env(self):
        """Render warthog environment"""
        #self.env.render()
        #time.sleep(0.05)
        pass

    def cmd_vel_cb(self, msg):
        """Ros command velocity callback"""
        x = self.env.pose[0]
        y = self.env.pose[1]
        th = self.env.pose[2]
        v = self.env.twist[0]
        w = self.env.twist[1]
        v_cmd = msg.linear.x
        w_cmd = msg.angular.z
        self.file_h.writelines(f"{x}, {y}, {th}, {v}, {w}, {v_cmd}, {w_cmd}\n")
        self.env.sim_warthog(msg.linear.x, msg.angular.z)

    def joy_cb(self, msg):
        """Ros command velocity callback"""
        global accel, brake, steer, mutex
        x = self.env.pose[0]
        y = self.env.pose[1]
        th = self.env.pose[2]
        v = self.env.twist[0]
        w = self.env.twist[1]
        mutex.acquire()
        accel = (1.0 - msg.axes[5]) / 2.0
        brake = (1.0 - msg.axes[2]) / 2.0
        steer = msg.axes[3]
        print(accel, brake, steer)
        mutex.release()
        #self.file_h.writelines(f"{x}, {y}, {th}, {v}, {w}, {accel}, {brake}\n")
        #self.env.sim_ranger(accel, brake, steer)


def main():
    """Ros node to start warthog simulation and collect data"""
    rospy.init_node("sim_remote_data_collector")
    data_collector = DataCollector()
    plt.pause(3)
    r = rospy.Rate(30)
    global accel, brake, steer, mutex
    while (not rospy.is_shutdown()):
        x = data_collector.env.pose[0]
        y = data_collector.env.pose[1]
        th = data_collector.env.pose[2]
        v = data_collector.env.twist[0]
        w = data_collector.env.twist[1]
        data_collector.file_h.writelines(
            f"{x}, {y}, {th}, {v}, {w}, {accel}, {brake}\n")
        time1 = time.time()
        mutex.acquire()
        data_collector.env.sim_ranger(accel, brake, steer)
        mutex.release()
        time2 = time.time()
        data_collector.env.render()
        delt = 0.03 - (time2 - time1)
        print("delt: ", delt)
        if delt >= 0:
            time.sleep(delt)
    data_collector.file_h.close()


if __name__ == '__main__':
    main()
