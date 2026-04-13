-- 切換資料庫
\connect arm_tree_db

-- 建立 nodes 表
CREATE TABLE IF NOT EXISTS nodes (
  id SERIAL PRIMARY KEY,
  key TEXT UNIQUE NOT NULL,
  joint_angles DOUBLE PRECISION[] NOT NULL,
  ee_pose DOUBLE PRECISION[] NOT NULL
);

-- 建立 edges 表
CREATE TABLE IF NOT EXISTS edges (
  id SERIAL PRIMARY KEY,
  from_node INTEGER REFERENCES nodes(id),
  to_node INTEGER REFERENCES nodes(id),
  direction TEXT NOT NULL
);

-- 將所有權轉給 arm_user（或給權限）
ALTER TABLE nodes OWNER TO arm_user;
ALTER TABLE edges OWNER TO arm_user;

-- 或者給 arm_user 所有存取權（如果你不想轉移所有權）
-- GRANT ALL PRIVILEGES ON TABLE nodes TO arm_user;
-- GRANT ALL PRIVILEGES ON TABLE edges TO arm_user;