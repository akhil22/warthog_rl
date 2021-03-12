from env.WarthogEnv import WarthogEnv
import rospy
import threading
import time
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt

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
        self.env = WarthogEnv(None)
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', 'warthog_velocity_controller/cmd_vel')
        self.out_file = rospy.get_param('~out_file_name', 'sim_remote_poses_delayed.csv')
        self.file_h = open(self.out_file, 'w')
        self.file_h.writelines(f"x,y,th,vel,w,v_cmd,w_cmd\n")
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_cb)
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

def main():
    """Ros node to start warthog simulation and collect data"""
    rospy.init_node("sim_remote_data_collector")
    data_collector = DataCollector()
    plt.pause(3)
    r = rospy.Rate(10)
    while(not rospy.is_shutdown()):
        #data_collector.env.render()
        r.sleep()
    data_collector.file_h.close()

if __name__ == '__main__':
    main()



