import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ScanRelayer(Node):
    def __init__(self):
        super().__init__('scan_relayer')
        
        # Subscribe to Unity's raw scan with drifted timestamp
        self.sub = self.create_subscription(
            LaserScan,
            '/scan_tmp',
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        
        # Publish to the standard /scan topic with synchronized container time
        self.pub = self.create_publisher(
            LaserScan,
            '/scan',
            rclpy.qos.qos_profile_sensor_data
        )
        self.get_logger().info('Scan Relayer started: /scan_tmp -> /scan (syncing timestamps to container time)')

    def scan_callback(self, msg: LaserScan):
        # Override the drifted Unity timestamp with the container's exact current time
        msg.header.stamp = self.get_clock().now().to_msg()
        # Publish the synced message
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ScanRelayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
