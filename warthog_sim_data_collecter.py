from env.WarthogEnv import WarthogEnv
import rospy
import threading
import time
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt

class DataCollector:
    def __init__(self):
        self.env = WarthogEnv(None)
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', 'warthog_velocity_controller/cmd_vel')
        self.out_file = rospy.get_param('~out_file_name', 'sim_remote_poses.csv')
        self.file_h = open(self.out_file, 'w')
        self.file_h.writelines(f"x,y,th,vel,w,v_cmd,w_cmd\n")
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_cb)
    def render_env(self):
        self.env.render()
        time.sleep(0.05)
    def cmd_vel_cb(self, msg):
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
    rospy.init_node("sim_remote_data_collector")
    data_collector = DataCollector()
    plt.pause(3)
    r = rospy.Rate(10)
    while(not rospy.is_shutdown()):
        data_collector.env.render()
        r.sleep()
    data_collector.file_h.close()

if __name__ == '__main__':
    main()



