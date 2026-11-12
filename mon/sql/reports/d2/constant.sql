-- Panels 34 D, 35 D: vector(2137483648); 46 B: the literal 10737418240 (a scalar, one unlabeled series)
SELECT unnest(range(getvariable('start_ms'), getvariable('end_ms') + 1, getvariable('step_ms'))) AS t,
       CAST(getvariable('value') AS DOUBLE) AS value;
