import psycopg2

class DBHelper:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="arm_tree_db", user="arm_user", password="Aa101201301401", host="host.docker.internal"
        )

    def get_or_create_node(self, key, angles, pose):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM nodes WHERE key = %s", (key,))
        result = cur.fetchone()
        if result:
            return result[0]
        cur.execute(
            "INSERT INTO nodes (key, joint_angles, ee_pose) VALUES (%s, %s, %s) RETURNING id",
            (key, angles, pose),
        )
        node_id = cur.fetchone()[0]
        self.conn.commit()
        return node_id

    def edge_exists(self, from_id, direction):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM edges WHERE from_node = %s AND direction = %s",
            (from_id, direction),
        )
        return cur.fetchone() is not None

    def insert_edge(self, from_id, to_id, direction):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO edges (from_node, to_node, direction) VALUES (%s, %s, %s)",
            (from_id, to_id, direction),
        )
        self.conn.commit()

    def get_node_id_by_key(self, key):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM nodes WHERE key = %s", (key,))
        result = cur.fetchone()
        return result[0] if result else None