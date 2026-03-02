-- Diagnóstico manual de errores de unicidad en Etapa 4.1
-- Uso sugerido:
--   sqlite3 data/state/rg_atp.sqlite < docs/stage4_1_sqlite_diagnostico.sql

.headers on
.mode column

.print '=== 0) Estado base ==='
SELECT sqlite_version() AS sqlite_version;

SELECT
  (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='relation_extractions') AS has_relation_extractions,
  (SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='ux_relation_extractions_unit_target_type') AS has_target_index;

.print '\n=== 1) SQL real del índice ux_relation_extractions_unit_target_type ==='
SELECT name, sql
FROM sqlite_master
WHERE type='index'
  AND name='ux_relation_extractions_unit_target_type';

.print '\n=== 2) Conteos de NULL/blank que suelen causar normalizaciones conflictivas ==='
SELECT
  COUNT(*) AS total_rows,
  SUM(CASE WHEN target_norm_key IS NULL THEN 1 ELSE 0 END) AS target_norm_key_null,
  SUM(CASE WHEN TRIM(COALESCE(target_norm_key,'')) = '' THEN 1 ELSE 0 END) AS target_norm_key_blank_or_null,
  SUM(CASE WHEN scope_detail IS NULL THEN 1 ELSE 0 END) AS scope_detail_null,
  SUM(CASE WHEN TRIM(COALESCE(scope_detail,'')) = '' THEN 1 ELSE 0 END) AS scope_detail_blank_or_null,
  SUM(CASE WHEN source_unit_id IS NULL THEN 1 ELSE 0 END) AS source_unit_id_null
FROM relation_extractions;

.print '\n=== 3) Posibles colisiones bajo clave LEGACY (doc,unit,target,type) ==='
SELECT
  source_doc_key,
  source_unit_id,
  COALESCE(target_norm_key,'') AS target_norm_key_norm,
  relation_type,
  COUNT(*) AS n
FROM relation_extractions
GROUP BY
  source_doc_key,
  source_unit_id,
  COALESCE(target_norm_key,''),
  relation_type
HAVING COUNT(*) > 1
ORDER BY n DESC, source_doc_key, source_unit_id
LIMIT 200;

.print '\n=== 4) Posibles colisiones bajo clave ACTUAL (incluye scope/scope_detail/method/extract_version) ==='
SELECT
  source_doc_key,
  source_unit_id,
  COALESCE(target_norm_key,'') AS target_norm_key_norm,
  relation_type,
  scope,
  COALESCE(scope_detail,'') AS scope_detail_norm,
  method,
  extract_version,
  COUNT(*) AS n
FROM relation_extractions
GROUP BY
  source_doc_key,
  source_unit_id,
  COALESCE(target_norm_key,''),
  relation_type,
  scope,
  COALESCE(scope_detail,''),
  method,
  extract_version
HAVING COUNT(*) > 1
ORDER BY n DESC, source_doc_key, source_unit_id
LIMIT 200;

.print '\n=== 5) Detalle de filas para grupos LEGACY duplicados ==='
WITH legacy_dups AS (
  SELECT
    source_doc_key,
    source_unit_id,
    COALESCE(target_norm_key,'') AS target_norm_key_norm,
    relation_type
  FROM relation_extractions
  GROUP BY
    source_doc_key,
    source_unit_id,
    COALESCE(target_norm_key,''),
    relation_type
  HAVING COUNT(*) > 1
)
SELECT
  re.relation_id,
  re.citation_id,
  re.link_id,
  re.source_doc_key,
  re.source_unit_id,
  re.source_unit_number,
  COALESCE(re.target_norm_key,'') AS target_norm_key_norm,
  re.relation_type,
  re.scope,
  COALESCE(re.scope_detail,'') AS scope_detail_norm,
  re.method,
  re.extract_version,
  re.confidence,
  re.created_at
FROM relation_extractions re
JOIN legacy_dups d
  ON d.source_doc_key = re.source_doc_key
 AND d.source_unit_id = re.source_unit_id
 AND d.target_norm_key_norm = COALESCE(re.target_norm_key,'')
 AND d.relation_type = re.relation_type
ORDER BY
  re.source_doc_key,
  re.source_unit_id,
  target_norm_key_norm,
  re.relation_type,
  re.confidence DESC,
  re.created_at DESC,
  re.relation_id DESC
LIMIT 500;

.print '\n=== 6) Verificación de estructura de relation_extractions ==='
PRAGMA table_info(relation_extractions);

.print '\n=== 7) Índices de relation_extractions ==='
PRAGMA index_list(relation_extractions);

.print '\n=== 8) SQL completo de todos los índices de relation_extractions ==='
SELECT name, sql
FROM sqlite_master
WHERE type='index'
  AND tbl_name='relation_extractions'
ORDER BY name;
