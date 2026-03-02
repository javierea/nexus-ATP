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

.print '\n=== 3) Casi duplicados ignorando scope_detail (base para canonización) ==='
SELECT
  source_doc_key,
  source_unit_id,
  COALESCE(target_norm_key,'') AS target_norm_key_norm,
  relation_type,
  scope,
  method,
  extract_version,
  COUNT(*) AS grp_n,
  SUM(CASE WHEN TRIM(COALESCE(scope_detail,'')) = '' THEN 1 ELSE 0 END) AS blank_scope_detail_n,
  GROUP_CONCAT(DISTINCT COALESCE(scope_detail,''), ' | ') AS scope_detail_variants
FROM relation_extractions
GROUP BY
  source_doc_key,
  source_unit_id,
  COALESCE(target_norm_key,''),
  relation_type,
  scope,
  method,
  extract_version
HAVING COUNT(*) > 1
ORDER BY grp_n DESC, source_doc_key, source_unit_id
LIMIT 300;

.print '\n=== 4) Ranking por grupo (rn, grp_n) para elegir candidata a conservar ==='
WITH ranked AS (
  SELECT
    re.relation_id,
    re.source_doc_key,
    re.source_unit_id,
    COALESCE(re.target_norm_key,'') AS target_norm_key_norm,
    re.relation_type,
    re.scope,
    COALESCE(re.scope_detail,'') AS scope_detail_norm,
    re.method,
    re.extract_version,
    re.confidence,
    re.created_at,
    COUNT(*) OVER (
      PARTITION BY
        re.source_doc_key,
        re.source_unit_id,
        COALESCE(re.target_norm_key,''),
        re.relation_type,
        re.scope,
        re.method,
        re.extract_version
    ) AS grp_n,
    ROW_NUMBER() OVER (
      PARTITION BY
        re.source_doc_key,
        re.source_unit_id,
        COALESCE(re.target_norm_key,''),
        re.relation_type,
        re.scope,
        re.method,
        re.extract_version
      ORDER BY re.confidence DESC, re.created_at DESC, re.relation_id DESC
    ) AS rn
  FROM relation_extractions re
)
SELECT
  relation_id,
  source_doc_key,
  source_unit_id,
  target_norm_key_norm,
  relation_type,
  scope,
  scope_detail_norm,
  method,
  extract_version,
  confidence,
  created_at,
  grp_n,
  rn
FROM ranked
WHERE grp_n > 1
ORDER BY grp_n DESC, source_doc_key, source_unit_id, rn
LIMIT 500;

.print '\n=== 5) Clasificación operativa de grupos ==='
WITH grouped AS (
  SELECT
    source_doc_key,
    source_unit_id,
    COALESCE(target_norm_key,'') AS target_norm_key_norm,
    relation_type,
    scope,
    method,
    extract_version,
    COUNT(*) AS grp_n,
    SUM(CASE WHEN TRIM(COALESCE(scope_detail,'')) = '' THEN 1 ELSE 0 END) AS blank_n,
    SUM(CASE WHEN scope = 'ARTICLE' AND UPPER(TRIM(COALESCE(scope_detail,''))) GLOB 'ART_*' THEN 1 ELSE 0 END) AS article_token_n,
    COUNT(DISTINCT UPPER(TRIM(COALESCE(scope_detail,'')))) AS scope_detail_variants_n
  FROM relation_extractions
  GROUP BY
    source_doc_key,
    source_unit_id,
    COALESCE(target_norm_key,''),
    relation_type,
    scope,
    method,
    extract_version
  HAVING COUNT(*) > 1
)
SELECT
  source_doc_key,
  source_unit_id,
  target_norm_key_norm,
  relation_type,
  scope,
  method,
  extract_version,
  grp_n,
  blank_n,
  article_token_n,
  scope_detail_variants_n,
  CASE
    WHEN scope = 'ARTICLE' AND article_token_n = grp_n AND scope_detail_variants_n > 1 THEN 'NO_TOCAR_ART_TOKEN'
    WHEN blank_n > 0 THEN 'REVISAR_VACIO_O_BLANCO'
    ELSE 'REVISAR_VARIANTE_TEXTO'
  END AS group_classification
FROM grouped
ORDER BY grp_n DESC, source_doc_key, source_unit_id
LIMIT 300;

.print '\n=== 6) Posibles colisiones bajo clave ACTUAL (incluye scope/scope_detail/method/extract_version) ==='
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

.print '\n=== 7) Verificación de estructura de relation_extractions ==='
PRAGMA table_info(relation_extractions);

.print '\n=== 8) Índices de relation_extractions ==='
PRAGMA index_list(relation_extractions);

.print '\n=== 9) SQL completo de todos los índices de relation_extractions ==='
SELECT name, sql
FROM sqlite_master
WHERE type='index'
  AND tbl_name='relation_extractions'
ORDER BY name;
