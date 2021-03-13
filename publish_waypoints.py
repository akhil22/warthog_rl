import rospy
import pandas as pd
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import sys


def main():
    rospy.init_node("waypoint_publisher")
    file_name = sys.argv[1]
    path_topic = rospy.get_param("~path_topic",
                                 '/local_planning/path/final_trajecotry')
    path_pub = rospy.Publisher(path_topic, Path, latch=True, queue_size=1)
    waypoint_path = Path()
    waypoint_path.header.frame_id = '/map'
    waypoint_path.header.stamp = rospy.get_rostime()

    df = pd.read_csv(file_name)
    print(df.keys())
    num_waypoints = len(df.index)
    print(df)
    for i in range(0, num_waypoints):
        current_pose = PoseStamped()
        current_pose.pose.position.x = df.iloc[i]["0.0"]
        current_pose.pose.position.y = df.iloc[i][" 0.0"]
        current_pose.pose.position.z = df.iloc[i][" 0.0.2"]
        waypoint_path.poses.append(current_pose)

    path_pub.publish(waypoint_path)
    rospy.spin()


if __name__ == '__main__':
    main()
