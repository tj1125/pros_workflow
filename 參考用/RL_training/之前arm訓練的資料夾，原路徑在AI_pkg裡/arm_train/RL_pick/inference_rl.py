# inference_rl.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from db_helper import DBHelper
import pickle
import numpy as np

class RLInference(Node):
    def __init__(self):
        super().__init__('rl_inference')
        self.db = DBHelper()
        self.edges = {}
        for nid in [row[0] for row in self._all_node_ids()]:
            self.edges[nid] = self.db.get_outgoing_edges(nid)

        # 載入已訓練好的 Q
        with open('qtable.pkl','rb') as f:
            self.Q = pickle.load(f)

        self.current = self.db.get_root_node()
        self.obs = None
        self.sub = self.create_subscription(
            Float32MultiArray, '/cubes_coordinates', self._obs_cb, 10)
        self.pub = self.create_publisher(String, '/arm_control_signal', 10)

    def _all_node_ids(self):
        cur = self.db.conn.cursor()
        cur.execute("SELECT id FROM nodes")
        return cur.fetchall()

    def _obs_cb(self, msg: Float32MultiArray):
        self.obs = np.array(msg.data[:3])
        self._decide_and_move()

    def _decide_and_move(self):
        acts = self.edges.get(self.current, [])
        if not acts:
            return
        # 選擇最高 Q
        best = max(acts, key=lambda a: self.Q.get((self.current,a[0]), -np.inf))
        dir, nxt = best
        self.pub.publish(String(data=dir))
        self.get_logger().info(f"[INFER] {self.current} -> {nxt} via {dir}")
        self.current = nxt

def main():
    rclpy.init()
    node = RLInference()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()