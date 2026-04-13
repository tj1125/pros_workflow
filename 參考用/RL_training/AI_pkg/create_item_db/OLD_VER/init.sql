-- 🚀 建立資料庫（由 superuser 建立，後面會把擁有權交給 item_in_house_user）
CREATE DATABASE item_in_house_db OWNER postgres;

-- 切換到 item_in_house_db
\connect item_in_house_db

-- 啟用 pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 建立新表，只留三個欄位：id、自動遞增；image_path (唯一)；content；embedding
CREATE TABLE IF NOT EXISTS object_data (
  id         SERIAL       PRIMARY KEY,
  image_path TEXT         UNIQUE NOT NULL,    -- 圖片路徑，必須唯一
  content    TEXT         NOT NULL,           -- 座標＋Caption＋image_path 的組合字串
  embedding  VECTOR(768)                      -- 存放 768 維向量
);

-- （以下把擁有權交給 item_in_house_user。若你已經有這個 user，請保留；否則先自行 CREATE USER）
ALTER DATABASE item_in_house_db OWNER TO item_in_house_user;
ALTER SCHEMA public            OWNER TO item_in_house_user;
ALTER TABLE  object_data       OWNER TO item_in_house_user;