import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from rl_model_wrapper import load_model_and_predict

class ArmPredictNode(Node):
    def __init__(self):
        super().__init__("arm_predict_node")
        self.subscription = self.create_subscription(
            Float32MultiArray, '/cubes_coordinates', self.listener_callback, 10
        )
        self.publisher = self.create_publisher(String, '/arm_control_signal', 10)
        self.model = load_model_and_predict("ppo_arm_model.zip")

    def listener_callback(self, msg):
        obs = list(msg.data)
        dx, dy, dz = obs[3] - obs[0], obs[4] - obs[1], obs[5] - obs[2]
        
        # ✅ 只用 [dx, dy, dz] 當 observation
        state = [dx, dy, dz]
        
        action = self.model(state)

        # 轉成 "0:i" or "0:k" 這種 arm_control_signal 格式
        joint = action // 2
        direction = "i" if action % 2 == 0 else "k"
        control_signal = f"{joint}:{direction}"
        
        msg_to_publish = String()
        msg_to_publish.data = control_signal
        self.publisher.publish(msg_to_publish)

        self.get_logger().info(f"✅ Sent control: {control_signal} | obs: {state}")

def main(args=None):
    rclpy.init(args=args)
    node = ArmPredictNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()