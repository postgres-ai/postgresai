-- Panel 41 A/B/C: irate(<metric>{matchers}[$__rate_interval])
--                   * on(datname) pgwatch_settings_numeric_value{datname=~"$db_name", setting_name="block_size"}
-- One-to-one matching on(datname): the result keeps ONLY the datname label (Prometheus/VM rule).
-- Both sides must have exactly one series per datname (else VM errors; here: not checked).
WITH a AS (
  SELECT coalesce(labels['datname'], '') AS datname, t, value
  FROM prom_irate(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), b AS (
  SELECT coalesce(labels['datname'], '') AS datname, t, value
  FROM prom_last('pgwatch_settings_numeric_value', [['datname','=~',getvariable('db_re')],['setting_name','=','block_size']],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
)
SELECT nullif(datname, '') AS datname, t, a.value * b.value AS value FROM a JOIN b USING (datname, t);
