-- === 建立使用者（若不存在）===
DO $$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles WHERE rolname = 'arm_user'
   ) THEN
      CREATE ROLE arm_user LOGIN PASSWORD 'Aa101201301401';
   END IF;
END
$$;

-- === ❗這行必須獨立在 DO 區塊外面執行 ===
-- 建立資料庫並設為 arm_user 擁有者（若不存在）
-- 必須手動確認是否已存在，否則會報錯
-- 所以建議在執行這個 .sql 前確認該 DB 不存在
CREATE DATABASE arm_tree_db OWNER arm_user;