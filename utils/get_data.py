#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
from std_msgs.msg import Float32
import math
import matplotlib.pyplot as plt

class LaserScanToNPZ:
    def __init__(self):
        rospy.init_node('laser_scan_to_npz')

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.vx_sub = rospy.Subscriber('/state/vx', Float32, self.vx_callback)
        self.vy_sub = rospy.Subscriber('/state/vy', Float32, self.vy_callback)
        self.w_sub = rospy.Subscriber('/state/w', Float32, self.w_callback)
        
        self.data = {'scan': [], 'vx': [], 'vy': [], 'w': [], 'x': []}
        self.curr_ranges = None
        self.vx = 0.
        self.vy = 0.
        self.w = 0.
        self.x = [0.,0.,0.]

    def scan_callback(self, msg):
        # Assuming LaserScan message has ranges field
        # msg.ranges = msg.ranges[::10]
        # msg.angle_increment = msg.angle_increment*10
        if self.curr_ranges is not None:
            diff = np.array(msg.ranges) - np.array(self.curr_ranges)
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges)//10)
            A = []
            b = []
            for i in range(len(angles)) :
                a = angles[i]
                # if np.abs(a) < 0.001 :
                #     print("r",msg.ranges[i-1],msg.ranges[i],msg.ranges[(i+1)%len(msg.ranges)])
                    # continue
                # else :
                theta = a + np.pi/2.
                _theta = theta - msg.angle_increment
                _pos = np.array([msg.ranges[i-1]*np.cos(_theta), msg.ranges[i-1]*np.sin(_theta)])
                theta_ = theta + msg.angle_increment
                ind = (i+1)%len(msg.ranges)
                if msg.ranges[ind]==0. or msg.ranges[i]==0. or msg.ranges[i-1]==0.:
                    continue
                if np.isnan(msg.ranges[ind]) or np.isnan(msg.ranges[i]) or np.isnan(msg.ranges[i-1]):
                    continue
                if self.curr_ranges[ind]==0. or self.curr_ranges[i]==0. or self.curr_ranges[i-1]==0.:
                    continue
                if np.isnan(self.curr_ranges[ind]) or np.isnan(self.curr_ranges[i]) or np.isnan(self.curr_ranges[i-1]):
                    continue
                if np.isinf(msg.ranges[ind]) or np.isinf(msg.ranges[i]) or np.isinf(msg.ranges[i-1]):
                    continue
                if np.isinf(self.curr_ranges[ind]) or np.isinf(self.curr_ranges[i]) or np.isinf(self.curr_ranges[i-1]):
                    continue
                pos_ = np.array([msg.ranges[ind]*np.cos(theta_), msg.ranges[ind]*np.sin(theta_)])
                alpha = np.arctan2(pos_[1] - _pos[1], pos_[0] - _pos[0])
                # print("a",alpha-theta)
                factor = 1./np.cos(alpha-np.pi/2.-theta)
                if np.abs(diff[i]) == 0. or np.abs(diff[i]) > 0.3 or np.isnan(diff[i]):
                    continue
                A.append([-np.sin(alpha)*factor, np.cos(alpha)*factor,msg.ranges[i]*(factor**2)*np.sin(alpha-np.pi/2.-theta)])
                b.append(diff[i])
            A = np.array(A)
            b = np.array(b)
            x = np.linalg.pinv(A).dot(b)
            self.x = x*7
            # print("M",A,b)
            # print(x*10)
            # print(diff)

        self.curr_ranges = msg.ranges
        
    def w_callback(self, msg):
        self.w = msg.data

    def vx_callback(self, msg):
        self.vx = msg.data
    
    def vy_callback(self, msg):
        self.vy = msg.data

    def save_npz(self, file_name):
        # print(self.x)
        t = np.arange(0, len(self.data['scan']), 1)*0.1
        plt.plot(t,self.data['vx'],label='vx')
        plt.plot(t,np.array(self.data['vx'])+np.array(self.data['x'])[:,1],label='vx predicted')
        plt.legend()
        plt.savefig("vx.png")
        plt.show()
        plt.plot(t,self.data['vy'],label='vy')
        plt.plot(t,np.array(self.data['vy'])+np.array(self.data['x'])[:,0],label='vy predicted')
        plt.legend()
        plt.savefig("vy.png")
        plt.show()
        plt.plot(t,self.data['w'],label='w')
        plt.plot(t,np.array(self.data['w'])+np.array(self.data['x'])[:,2],label='w predicted')
        plt.legend()
        plt.savefig("w.png")
        plt.show()
        np.savez(file_name, **self.data)

    def spin(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.curr_ranges is not None:
                # print(self.vx, self.vy)
                self.data['scan'].append(self.curr_ranges)
                self.data['vx'].append(self.vx)
                self.data['vy'].append(self.vy)
                self.data['w'].append(self.w)
                self.data['x'].append(self.x)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LaserScanToNPZ()
        file_name = rospy.get_param("~file_name", "laser_data.npz")  # Get file name from parameter server
        print("Started")
        node.spin()
        print("Saving data to %s" % file_name)
        node.save_npz(file_name)
    except rospy.ROSInterruptException:
        pass
