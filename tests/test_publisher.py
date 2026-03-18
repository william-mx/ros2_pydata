#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Import message types
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, Imu, MagneticField
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

# Import your conversion functions
from ros2_pydata.conversions import (
    np_to_image, np_to_compressedimage, np_to_pointcloud, 
    np_to_imu, np_to_pose, np_to_path, np_to_magneticfield,
    to_ackermann
)

class AllMessagePublisher(Node):
    def __init__(self):
        super().__init__('multi_message_publisher')
        
        # Create QoS profile
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        
        # Create publishers for different message types
        self.pubs = {
            'image': self.create_publisher(Image, 'image', qos),
            'compressed_image': self.create_publisher(CompressedImage, 'compressed_image', qos),
            'point_cloud': self.create_publisher(PointCloud2, 'pointcloud', qos),
            'imu': self.create_publisher(Imu, 'imu', qos),
            'pose': self.create_publisher(PoseStamped, 'pose', qos),
            'path': self.create_publisher(Path, 'path', qos),
            'magnetic_field': self.create_publisher(MagneticField, 'magnetic_field', qos),
            'ackermann': self.create_publisher(AckermannDriveStamped, 'ackermann', qos),
        }
        
        # Timer for publishing every 5 seconds
        self.timer = self.create_timer(5.0, self.publish_all_messages)
        
        self.get_logger().info('All-message publisher node started - publishing all message types every 5 seconds')
    
    def publish_all_messages(self):
        # Generate and publish all message types
        
        # Image
        image_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_msg = np_to_image(image_array)
        self.pubs['image'].publish(image_msg)
        
        # Compressed Image
        compressed_msg = np_to_compressedimage(image_array)
        self.pubs['compressed_image'].publish(compressed_msg)
        
        # PointCloud2
        points = np.random.rand(100, 3) * 10  # Random points in 10x10x10 volume
        pointcloud_msg = np_to_pointcloud(points)
        self.pubs['point_cloud'].publish(pointcloud_msg)
        
        # IMU
        imu_data = np.random.randn(6)  # Random accelerations and angular velocities
        imu_msg = np_to_imu(imu_data)
        self.pubs['imu'].publish(imu_msg)
        
        # Pose
        position = np.random.rand(3) * 10
        yaw = np.random.rand() * 2 * np.pi - np.pi  # Random yaw between -pi and pi
        pose_msg = np_to_pose(position, yaw)
        self.pubs['pose'].publish(pose_msg)
        
        # Path
        waypoints = np.random.rand(10, 3) * 10  # 10 random points with x, y, yaw
        waypoints[:, 2] = np.random.rand(10) * 2 * np.pi - np.pi  # Random yaw angles
        path_msg = np_to_path(waypoints)
        self.pubs['path'].publish(path_msg)
        
        # Magnetic Field
        mag_data = np.random.randn(3) * 50  # Random magnetic field values in μT
        mag_msg = np_to_magneticfield(mag_data)
        self.pubs['magnetic_field'].publish(mag_msg)
        
        # Ackermann Drive
        speed = np.random.rand() * 5.0  # Random speed between 0 and 5 m/s
        steering_angle = (np.random.rand() * 0.6 - 0.3)  # Random steering angle between -0.3 and 0.3 rad
        ackermann_msg = to_ackermann(speed, steering_angle)
        self.pubs['ackermann'].publish(ackermann_msg)
        
        self.get_logger().info('Published all message types')

def main(args=None):
    rclpy.init(args=args)
    node = AllMessagePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()