from env.WarthogEnvAirSim import WarthogEnv
import threading
import time
from matplotlib import pyplot as plt
import pygame
pygame.init()
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
        """Initialize Data collector with gym environment
        command velocity topic and output file name
        """
        self.env = WarthogEnv(None)
        self.out_file = 'AirSimPose.csv' 
        self.file_h = open(self.out_file, 'w')
        self.file_h.writelines(f"x,y,th,vel,w,v_cmd,w_cmd\n")
        self.joystick = pygame.joystick.Joystick(0)
        self.prev_time = None
    def render_env(self):
        """Render warthog environment"""
        #self.env.render()
        #time.sleep(0.05)
        pass
    def cmd_vel_cb(self, v_cmd, w_cmd):
        """Ros command velocity callback"""
        x = self.env.pose[0]
        y = self.env.pose[1]
        th = self.env.pose[2]
        v = self.env.twist[0]
        w = self.env.twist[1]
        #v_cmd = msg.linear.x
        #w_cmd = msg.angular.z
        self.file_h.writelines(f"{x}, {y}, {th}, {v}, {w}, {v_cmd}, {w_cmd}\n")
        curr_time = time.time()
        if (curr_time - self.prev_time) >= 0.058:
            self.env.sim_warthog(v_cmd, w_cmd)
            self.prev_time = curr_time
        #self.env.render()

def main():
    """Ros node to start warthog simulation and collect data"""
    data_collector = DataCollector()
    #plt.pause(3)
    data_collector.env.render()
    done = False
    data_collector.prev_time = time.time()
    while(not done):
        a0  = data_collector.joystick.get_axis(1)
        a1  = data_collector.joystick.get_axis(2)
        v = -a0*5.0
        w = -a1*2.0
        data_collector.cmd_vel_cb(v, w)
        print(v,w)
        time.sleep(0.01)

if __name__ == '__main__':
    main()



