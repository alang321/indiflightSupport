import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
import pandas as pd
from std_msgs.msg import Header

class CsvToRos2Publisher(Node):
    def __init__(self, df):
        super().__init__('csv_to_ros2_publisher')
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps', 10)
        self.df = df
        self.row_index = 0
        self.timer = self.create_timer(0.1, self.publish_row)  # Adjust rate as needed

    def publish_row(self):
        if self.row_index >= len(self.df):
            self.get_logger().info("All rows published")
            self.timer.cancel()
            return

        # Convert timestamp to ROS 2 time
        timestamp = self.df.iloc[self.row_index]['timestamp']

        # Populate IMU Message
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()  # Adjust timestamp if needed
        # Example for IMU data, update with your actual CSV columns
        imu_msg.angular_velocity.x = self.df.iloc[self.row_index]['a_v_x']
        imu_msg.angular_velocity.y = self.df.iloc[self.row_index]['a_v_y']
        imu_msg.angular_velocity.z = self.df.iloc[self.row_index]['a_v_z']

        self.imu_pub.publish(imu_msg)

        # Populate GPS Message
        gps_msg = NavSatFix()
        gps_msg.header = Header()
        gps_msg.header.stamp = imu_msg.header.stamp  # Synchronize timestamps
        gps_msg.latitude = self.df.iloc[self.row_index]['latitude']
        gps_msg.longitude = self.df.iloc[self.row_index]['longitude']
        gps_msg.altitude = self.df.iloc[self.row_index]['altitude']

        self.gps_pub.publish(gps_msg)

        # Move to the next row
        self.row_index += 1

def main():
    # Load your CSV file
    df = pd.read_csv('data.csv')

    rclpy.init()
    csv_publisher = CsvToRos2Publisher(df)
    rclpy.spin(csv_publisher)
    csv_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()