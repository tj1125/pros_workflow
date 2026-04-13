# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray, String
# import json

# class AI_Cube_Node(Node):
#     def __init__(self):
#         super().__init__('ai_cube_node')
#         self.subscription = self.create_subscription(
#             Float32MultiArray, '/cubes_coordinates', self.listener_callback, 10)
#         self.publisher = self.create_publisher(String, '/cubes_action', 10)
#         self.cube_positions = None

#     def listener_callback(self, msg):
#         self.cube_positions = msg.data  # [Cube1_x, Cube1_y, Cube1_z, Cube2_x, Cube2_y, Cube2_z]
#         action = self.compute_action(self.cube_positions)
#         self.send_action(action)

#     def compute_action(self, positions):
#         cube1_x, cube1_y, cube1_z, cube2_x, cube2_y, cube2_z = positions
#         # print(f"Cube1: ({cube1_x}, {cube1_y}, {cube1_z}), Cube2: ({cube2_x}, {cube2_y}, {cube2_z})")

#         # 計算每個軸向的差距
#         delta_x = cube2_x - cube1_x
#         delta_y = cube2_y - cube1_y
#         delta_z = cube2_z - cube1_z

#         # 計算三維距離
#         distance = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5

#         # 每次移動的最大步伐
#         step_size = 0.1

#         if distance > 0:
#             move_x = (delta_x / distance) * step_size
#             move_y = (delta_y / distance) * step_size
#             move_z = (delta_z / distance) * step_size
#         else:
#             move_x = 0.0
#             move_y = 0.0
#             move_z = 0.0

#         return {
#             "moveX": move_x,
#             "moveY": move_y,
#             "moveZ": move_z,
#             "isTraining": True
#         }

#     def send_action(self, action):
#         msg = String()
#         msg.data = json.dumps(action)
#         self.publisher.publish(msg)

# # 啟動節點
# print("AI_Cube_Node is running...")
# rclpy.init()
# node = AI_Cube_Node()
# rclpy.spin(node)
# rclpy.shutdown()
# print("AI_Cube_Node is stopped...")