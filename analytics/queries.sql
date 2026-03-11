-- =============================================================================
-- LitLens Analytics SQL Queries
-- =============================================================================
-- Designed for PostgreSQL. Run via psql or any async driver connected to the
-- litlens database.  All queries are safe for repeated execution.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Query volume over time (last 30 days, grouped by day)
-- -----------------------------------------------------------------------------
SELECT
    date_trunc('day', created_at)::date                                AS day,
    COUNT(*)                                                           AS queries,
    ROUND(AVG(latency_ms)::numeric, 1)                                 AS avg_latency_ms,
    ROUND(AVG(papers_retrieved)::numeric, 1)                           AS avg_papers,
    SUM(CASE WHEN contradictions_found > 0 THEN 1 ELSE 0 END)         AS queries_with_contradictions
FROM queries
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY day
ORDER BY day;


-- -----------------------------------------------------------------------------
-- Top 50 query topic words by frequency (excludes common stop words)
-- -----------------------------------------------------------------------------
SELECT
    word,
    COUNT(*) AS frequency
FROM queries,
    unnest(string_to_array(lower(raw_query), ' ')) AS word
WHERE
    length(word) > 4
    AND word NOT IN (
        'about', 'after', 'against', 'before', 'between', 'comparison',
        'disease', 'effect', 'effects', 'evidence', 'following', 'human',
        'patients', 'recent', 'study', 'therapy', 'treatment', 'versus',
        'which', 'these', 'their', 'there', 'where', 'other', 'would',
        'could', 'should', 'using', 'based', 'clinical', 'management'
    )
GROUP BY word
ORDER BY frequency DESC
LIMIT 50;


-- -----------------------------------------------------------------------------
-- Latency distribution by query intent
-- -----------------------------------------------------------------------------
SELECT
    intent,
    COUNT(*)                                                               AS n,
    ROUND(AVG(latency_ms)::numeric, 1)                                     AS avg_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms)::numeric     AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric     AS p95_ms,
    MAX(latency_ms)                                                        AS max_ms
FROM queries
WHERE intent IS NOT NULL
  AND latency_ms IS NOT NULL
GROUP BY intent
ORDER BY avg_ms DESC;


-- -----------------------------------------------------------------------------
-- Synthesis quality (faithfulness score) by query intent
-- -----------------------------------------------------------------------------
SELECT
    intent,
    ROUND(AVG(faithfulness)::numeric, 3)                                   AS avg_faithfulness,
    COUNT(*) FILTER (WHERE faithfulness >= 0.75)                           AS above_threshold,
    COUNT(*)                                                               AS total,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE faithfulness >= 0.75) / COUNT(*),
        1
    )                                                                      AS pct_passing
FROM queries
WHERE faithfulness IS NOT NULL
  AND intent IS NOT NULL
GROUP BY intent
ORDER BY avg_faithfulness DESC;


-- -----------------------------------------------------------------------------
-- Contradiction detection rate over time (weekly)
-- -----------------------------------------------------------------------------
SELECT
    date_trunc('week', created_at)::date                                   AS week,
    COUNT(*)                                                               AS total_queries,
    SUM(CASE WHEN contradictions_found > 0 THEN 1 ELSE 0 END)             AS queries_with_contradictions,
    ROUND(
        100.0 * SUM(CASE WHEN contradictions_found > 0 THEN 1 ELSE 0 END) / COUNT(*),
        1
    )                                                                      AS contradiction_rate_pct
FROM queries
WHERE contradictions_found IS NOT NULL
GROUP BY week
ORDER BY week;


-- -----------------------------------------------------------------------------
-- User feedback summary by intent
-- -----------------------------------------------------------------------------
SELECT
    q.intent,
    COUNT(f.id)                                                            AS feedback_count,
    ROUND(AVG(f.rating)::numeric, 2)                                       AS avg_rating,
    COUNT(*) FILTER (WHERE f.rating >= 4)                                  AS positive,
    COUNT(*) FILTER (WHERE f.rating <= 2)                                  AS negative
FROM query_feedback f
JOIN queries q ON q.id = f.query_id
GROUP BY q.intent
ORDER BY avg_rating DESC;
