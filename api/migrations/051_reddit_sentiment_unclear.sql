-- 051_reddit_sentiment_unclear.sql
--
-- Sprint 8 audit polish - add the "unclear" sentiment value to the
-- scan_reddit_threads CHECK constraint.
--
-- Background : Haiku sometimes can't tell from the LLM citation snippets
-- alone whether the sentiment is positive / negative / neutral / mixed,
-- because the snippets are too thin (often just "[Source: reddit.com]"
-- with no body text from the discussion). The original v1 collapsed
-- this case into "neutral", which is misleading - "neutral" should
-- mean "factual, no opinion expressed", not "we couldn't tell."
--
-- New label "unclear" makes the distinction explicit ; the UI can chip
-- it differently (gray-italic) so the user understands the data is
-- thin, not that the thread is uncontroversial.

ALTER TABLE scan_reddit_threads
  DROP CONSTRAINT IF EXISTS rt_sentiment_values;

ALTER TABLE scan_reddit_threads
  ADD CONSTRAINT rt_sentiment_values CHECK (
    sentiment IS NULL OR sentiment IN ('positive', 'negative', 'neutral', 'mixed', 'unclear')
  );
