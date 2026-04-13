# db_helper.py
import psycopg2

class DBHelper:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="arm_tree_db",
            user="arm_user",
            password="Aa101201301401",
            host="host.docker.internal"
        )

    def get_or_create_node(self, key, angles, pose):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM nodes WHERE key=%s", (key,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO nodes(key,joint_angles,ee_pose) VALUES (%s,%s,%s) RETURNING id",
            (key, angles, pose),
        )
        node_id = cur.fetchone()[0]
        self.conn.commit()
        return node_id

    def edge_exists(self, from_id, direction):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM edges WHERE from_node=%s AND direction=%s",
            (from_id, direction),
        )
        return cur.fetchone() is not None

    def insert_edge(self, from_id, to_id, direction):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO edges(from_node,to_node,direction) VALUES (%s,%s,%s)",
            (from_id, to_id, direction),
        )
        self.conn.commit()

    def get_outgoing_edges(self, from_id):
        """
        回傳 list of (direction:str, to_node:int)
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT direction,to_node FROM edges WHERE from_node=%s",
            (from_id,)
        )
        return cur.fetchall()

    def get_root_node(self):
        """
        假設第 1 筆 node 是 root (或自行改用你的 root_id)
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM nodes ORDER BY id LIMIT 1")
        return cur.fetchone()[0]
    
    def get_next_node(self, from_id, direction):
        cur = self.conn.cursor()
        cur.execute(
        "SELECT to_node FROM edges WHERE from_node=%s AND direction=%s",
        (from_id, direction)
        )
        row = cur.fetchone()
        return row[0] if row else from_id