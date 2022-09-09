#from env.WarthogEnv import WarthogEnv
import rospy
import threading
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from matplotlib import pyplot as plt
from pyquaternion import Quaternion as qut
import message_filters
import numpy as np

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
        #self.env = WarthogEnv(None)
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', 'warthog_velocity_controller/cmd_vel')
        #self.odom_topic = rospy.get_param('~odom_topic', 'odometry/filtered_map')
        self.odom_topic = rospy.get_param('~odom_topic', 'odometry/filtered')
        self.gps_odom_topic = rospy.get_param('~map_odom_topic', 'odometry/filtered2')
        #self.odom_topic = rospy.get_param('~odom_topic', 'warthog_velocity_controller/odom')
        # self.out_file = rospy.get_param('~out_file_name', 'real_remote_poses_ext_war_gps2.csv')
        self.out_file = rospy.get_param('~out_file_name', 'new_hanger_grass1.csv')
        self.file_h = open(self.out_file, 'w')
        self.file_h.writelines("x,y,th,vel,w,v_cmd,w_cmd\n")
        self.cmd_vel_sub = message_filters.Subscriber(self.cmd_vel_topic, Twist)
        self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        self.gps_odom_sub = message_filters.Subscriber(self.gps_odom_topic, Odometry)
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.cmd_vel_sub, self.odom_sub],10, 1, allow_headerless=True)
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.cmd_vel_sub, self.odom_sub, self.gps_odom_sub],10, 1, allow_headerless=True)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cmd_vel_sub, self.odom_sub, self.gps_odom_sub],10, 1, allow_headerless=True)
        #self.cmd_odom_cb= None 
        #self.ts.registerCallback(self.cmd_odom_cb)

        #rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_cb)
    def render_env(self):
        """Render warthog environment"""
        #self.env.render()
        #time.sleep(0.05)
        pass
    def cmd_odom_cb(self, cmd_msg, odom_msg):
        """Ros command velocity callback"""
        #print(cmd_msg, odom_msg)
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        temp_y = odom_msg.pose.pose.orientation.z
        temp_x = odom_msg.pose.pose.orientation.w
        quat = (temp_x, 0, 0, temp_y) 
        myqut = qut(quat)
        th = myqut.radians*np.sign(temp_y)
        # y = self.env.pose[1]
        # th = self.env.pose[2]
        v = odom_msg.twist.twist.linear.x 
        w = odom_msg.twist.twist.angular.z 
        # w = self.env.twist[1]
        v_cmd = cmd_msg.linear.x
        w_cmd = cmd_msg.angular.z
        self.file_h.writelines(str(x)+','+str(y)+','+ str(th) +','+ str(v) +','+ str(w) +','+ str(v_cmd) +','+ str(w_cmd)+'\n')
        # w_cmd = msg.angular.z
        #self.file_h.writelines(f"{x}, {y}, {th}, {v}, {w}, {v_cmd}, {w_cmd}\n")
        #print(cmd_msg, odom_msg)
    def gps_cmd_odom_cb(self, cmd_msg, odom_msg, gps_msg):
        """Ros command velocity callback"""
        #print(cmd_msg, odom_msg)
        x = gps_msg.pose.pose.position.x
        y = gps_msg.pose.pose.position.y
        temp_y = gps_msg.pose.pose.orientation.z
        temp_x = gps_msg.pose.pose.orientation.w
        quat = (temp_x, 0, 0, temp_y) 
        myqut = qut(quat)
        th = myqut.radians*np.sign(temp_y)
        # y = self.env.pose[1]
        # th = self.env.pose[2]
        v = odom_msg.twist.twist.linear.x 
        w = odom_msg.twist.twist.angular.z 
        # w = self.env.twist[1]
        v_cmd = cmd_msg.linear.x
        w_cmd = cmd_msg.angular.z
        self.file_h.writelines(str(x)+','+str(y)+','+ str(th) +','+ str(v) +','+ str(w) +','+ str(v_cmd) +','+ str(w_cmd)+'\n')
        # w_cmd = msg.angular.z
        #self.file_h.writelines(f"{x}, {y}, {th}, {v}, {w}, {v_cmd}, {w_cmd}\n")
    def cmd_vel_cb(self, msg):
        """Ros command velocity callback"""
        pass
        # x = self.env.pose[0]
        # y = self.env.pose[1]
        # th = self.env.pose[2]
        # v = self.env.twist[0]
        # w = self.env.twist[1]
        # v_cmd = msg.linear.x
        # w_cmd = msg.angular.z
        # #self.file_h.writelines(f"{x}, {y}, {th}, {v}, {w}, {v_cmd}, {w_cmd}\n")
        # self.env.sim_warthog(msg.linear.x, msg.angular.z)

def main():
    """Ros node to start warthog simulation and collect data"""
    rospy.init_node("sim_remote_data_collector")
    data_collector = DataCollector()
    #data_collector.ts.registerCallback(data_collector.cmd_odom_cb)
    data_collector.ts.registerCallback(data_collector.gps_cmd_odom_cb)
    plt.pause(3)
    r = rospy.Rate(10)
    while(not rospy.is_shutdown()):
        #data_collector.env.render()
        r.sleep()
    data_collector.file_h.close()

if __name__ == '__main__':
    main()



