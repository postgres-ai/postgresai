/**
 * Generate human-readable summaries from checkup report JSON.
 * Used for default CLI output without requiring API calls.
 *
 * NOTE: This file uses `any` types for report structures to maintain flexibility
 * with the dynamic JSON schema from the API. Future improvement: define proper
 * TypeScript interfaces for report structures based on schema files.
 */

export interface CheckSummary {
  status: 'ok' | 'warning' | 'info';
  message: string;
}

/**
 * Extract summary information from a checkup report.
 * Parses the JSON structure to extract key metrics for CLI display.
 */
export function generateCheckSummary(checkId: string, report: any): CheckSummary {
  const nodeIds = Object.keys(report.results || {});
  if (nodeIds.length === 0) {
    return { status: 'info', message: 'No data' };
  }

  // Take first node for summary (most deployments use single node)
  const nodeData = report.results[nodeIds[0]];

  switch (checkId) {
    // Index health checks
    case 'H001': return summarizeH001(nodeData);
    case 'H002': return summarizeH002(nodeData);
    case 'H004': return summarizeH004(nodeData);
    // Version checks
    case 'A002': return summarizeA002(nodeData);
    case 'A013': return summarizeA013(nodeData);
    // Settings checks
    case 'A003': return summarizeA003(nodeData);
    case 'A004': return summarizeA004(nodeData);
    case 'A007': return summarizeA007(nodeData);
    case 'D001': return summarizeD001(nodeData);
    case 'D004': return summarizeD004(nodeData);
    case 'F001': return summarizeF001(nodeData);
    case 'F002': return summarizeF002(nodeData);
    case 'F003': return summarizeF003(nodeData);
    case 'F004': return summarizeBloat(nodeData, 'table');
    case 'F005': return summarizeBloat(nodeData, 'index');
    case 'F009': return summarizeF009(nodeData);
    case 'G001': return summarizeG001(nodeData);
    case 'G003': return summarizeG003(nodeData);
    default:
      return { status: 'info', message: 'Check completed' };
  }
}

function summarizeA003(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  const settingsCount = Object.keys(data).length;

  if (settingsCount === 0) {
    return { status: 'info', message: 'No settings found' };
  }

  return {
    status: 'info',
    message: `${settingsCount} setting${settingsCount > 1 ? 's' : ''} collected`
  };
}

function summarizeA004(nodeData: any): CheckSummary {
  const data = nodeData?.data;
  if (!data) {
    return { status: 'info', message: 'Cluster information collected' };
  }

  const dbCount = Object.keys(data.database_sizes || {}).length;
  if (dbCount > 0) {
    return { status: 'info', message: `${dbCount} database${dbCount > 1 ? 's' : ''} analyzed` };
  }

  return { status: 'info', message: 'Cluster information collected' };
}

function summarizeA007(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  const alteredCount = Object.keys(data).length;

  if (alteredCount === 0) {
    return { status: 'ok', message: 'No altered settings' };
  }

  return {
    status: 'info',
    message: `${alteredCount} setting${alteredCount > 1 ? 's' : ''} altered from defaults`
  };
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GiB`;
}

function summarizeH001(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  let totalCount = 0;
  let totalSize = 0;

  // Aggregate across all databases
  for (const dbData of Object.values(data)) {
    const dbEntry = dbData as any;
    totalCount += dbEntry.total_count || 0;
    totalSize += dbEntry.total_size_bytes || 0;
  }

  if (totalCount === 0) {
    return { status: 'ok', message: 'No invalid indexes' };
  }

  return {
    status: 'warning',
    message: `Found ${totalCount} invalid index${totalCount > 1 ? 'es' : ''} (${formatBytes(totalSize)})`
  };
}

function summarizeH002(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  let totalCount = 0;
  let totalSize = 0;

  // Aggregate across all databases
  for (const dbData of Object.values(data)) {
    const dbEntry = dbData as any;
    totalCount += dbEntry.total_count || 0;
    totalSize += dbEntry.total_size_bytes || 0;
  }

  if (totalCount === 0) {
    return { status: 'ok', message: 'All indexes utilized' };
  }

  return {
    status: 'warning',
    message: `Found ${totalCount} unused index${totalCount > 1 ? 'es' : ''} (${formatBytes(totalSize)})`
  };
}

function summarizeH004(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  let totalCount = 0;
  let totalSize = 0;

  // Aggregate across all databases
  for (const dbData of Object.values(data)) {
    const dbEntry = dbData as any;
    totalCount += dbEntry.total_count || 0;
    totalSize += dbEntry.total_size_bytes || 0;
  }

  if (totalCount === 0) {
    return { status: 'ok', message: 'No redundant indexes' };
  }

  return {
    status: 'warning',
    message: `Found ${totalCount} redundant index${totalCount > 1 ? 'es' : ''} (${formatBytes(totalSize)})`
  };
}

function summarizeA002(nodeData: any): CheckSummary {
  // A002 stores version in data.version (not postgres_version)
  const ver = nodeData?.data?.version;
  if (!ver) {
    return { status: 'info', message: 'Version checked' };
  }

  const major = parseInt(ver.server_major_ver, 10);

  // PostgreSQL 17 is current (as of early 2025)
  if (major >= 17) {
    return { status: 'ok', message: `PostgreSQL ${major}` };
  }

  if (major >= 15) {
    return { status: 'info', message: `PostgreSQL ${major}` };
  }

  return {
    status: 'warning',
    message: `PostgreSQL ${major} (consider upgrading)`
  };
}

function summarizeA013(nodeData: any): CheckSummary {
  // A013 stores version in data.version (not postgres_version)
  const ver = nodeData?.data?.version;
  if (!ver) {
    return { status: 'info', message: 'Minor version checked' };
  }

  const current = ver.version || '';
  return {
    status: 'info',
    message: `Version ${current}`
  };
}

function summarizeD001(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  const settingsCount = Object.keys(data).length;

  if (settingsCount === 0) {
    return { status: 'info', message: 'No logging settings found' };
  }

  return {
    status: 'info',
    message: `${settingsCount} logging setting${settingsCount > 1 ? 's' : ''} collected`
  };
}

function summarizeD004(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  const settingsCount = Object.keys(data).length;

  if (settingsCount === 0) {
    return { status: 'info', message: 'No pg_stat_statements settings found' };
  }

  return {
    status: 'info',
    message: `${settingsCount} pg_stat_statements setting${settingsCount > 1 ? 's' : ''} collected`
  };
}

function summarizeF001(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  const settingsCount = Object.keys(data).length;
  const analysis = nodeData?.settings_analysis;
  const fired = analysis?.rules_fired || [];

  // Map the highest fired rule severity to a CLI status. CRITICAL/WARNING are
  // actionable (warning); NOTICE/INFO are informational.
  if (fired.length > 0) {
    const severity: string | null = analysis?.severity ?? null;
    const summarize = () => {
      const critical = fired.filter((r: any) => r.severity === 'CRITICAL').length;
      const warning = fired.filter((r: any) => r.severity === 'WARNING').length;
      const parts: string[] = [];
      if (critical > 0) parts.push(`${critical} critical`);
      if (warning > 0) parts.push(`${warning} warning`);
      const other = fired.length - critical - warning;
      if (other > 0) parts.push(`${other} advisory`);
      return `Autovacuum config: ${parts.join(', ')} finding${fired.length > 1 ? 's' : ''}`;
    };

    if (severity === 'CRITICAL' || severity === 'WARNING') {
      return { status: 'warning', message: summarize() };
    }
    return { status: 'info', message: summarize() };
  }

  if (settingsCount === 0) {
    return { status: 'info', message: 'No autovacuum settings found' };
  }

  return {
    status: 'ok',
    message: `Autovacuum config looks healthy (${settingsCount} setting${settingsCount > 1 ? 's' : ''} analyzed)`
  };
}

function summarizeF002(nodeData: any): CheckSummary {
  if (nodeData?.data?.settings_available === false) {
    return { status: 'info', message: 'Wraparound settings unavailable' };
  }
  const severity = nodeData?.data?.severity;
  const databases = nodeData?.data?.databases?.length || 0;
  const tables = nodeData?.data?.tables || [];
  const offenders = tables.filter((table: any) =>
    table?.xid?.severity !== 'info' || table?.multixact?.severity !== 'info'
  ).length;

  if (!severity) return { status: 'info', message: 'No wraparound data' };
  if (severity === 'info') {
    return { status: 'ok', message: `${databases} database${databases === 1 ? '' : 's'} below wraparound thresholds` };
  }
  return {
    status: 'warning',
    message: `${severity.toUpperCase()} wraparound risk${offenders ? `; ${offenders} table${offenders === 1 ? '' : 's'} affected` : ''}`,
  };
}

function summarizeF003(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  let flaggedCount = 0;
  let disabledCount = 0;

  // Aggregate across all databases. Only non-tiny disabled-autovacuum tables
  // (autovacuum_disabled_flagged_count) trigger a warning - tiny tables with
  // autovacuum off are common and not worth alerting on.
  for (const dbData of Object.values(data)) {
    const dbEntry = dbData as any;
    flaggedCount += dbEntry.flagged_count || 0;
    disabledCount += dbEntry.autovacuum_disabled_flagged_count || 0;
  }

  if (flaggedCount === 0 && disabledCount === 0) {
    return { status: 'ok', message: 'No significant dead tuple accumulation' };
  }

  const parts: string[] = [];
  if (flaggedCount > 0) {
    parts.push(`${flaggedCount} table${flaggedCount > 1 ? 's' : ''} with excessive dead tuples`);
  }
  if (disabledCount > 0) {
    parts.push(`${disabledCount} table${disabledCount > 1 ? 's' : ''} with autovacuum disabled`);
  }

  return { status: 'warning', message: parts.join(', ') };
}

function summarizeF009(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  if (data.skipped) {
    return { status: 'info', message: data.skip_reason || 'Primary-only check skipped on replica' };
  }
  const severity = String(data.severity || 'OK');
  const age = Math.max(Number(data.data_horizon_age_tx || 0), Number(data.catalog_horizon_age_tx || 0));
  if (severity === 'OK') {
    return { status: 'ok', message: `Xmin horizons healthy (oldest holder: ${age} transactions)` };
  }
  const holder = data.dominant_holder?.source ? `; dominant holder: ${data.dominant_holder.source}` : '';
  const activityAgeSeconds = Number(data.components?.pg_stat_activity?.top_blocker?.xact_age_seconds || 0);
  if (
    severity === 'NOTICE'
    && data.data_horizon_severity === 'OK'
    && data.catalog_horizon_severity === 'OK'
    && activityAgeSeconds > 0
  ) {
    return {
      status: 'info',
      message: `NOTICE: transaction open for ${activityAgeSeconds} seconds${holder}`,
    };
  }
  return {
    status: severity === 'NOTICE' ? 'info' : 'warning',
    message: `${severity}: xmin horizon age ${age} transactions${holder}`,
  };
}

function summarizeBloat(nodeData: any, kind: 'table' | 'index'): CheckSummary {
  const data = nodeData?.data || {};
  let totalCount = 0;

  for (const dbData of Object.values(data)) {
    const dbEntry = dbData as any;
    if (dbEntry?.status?.ok === false) {
      const reason = String(dbEntry.status.reason || 'query_error').replaceAll('_', ' ');
      return { status: 'warning', message: `Bloat estimate degraded: ${reason}` };
    }
    totalCount += dbEntry?.total_count || 0;
  }

  if (totalCount === 0) {
    return { status: 'ok', message: `No bloated ${kind}${kind === 'index' ? 'es' : 's'} found` };
  }

  return {
    status: 'warning',
    message: `Found ${totalCount} bloated ${kind}${totalCount === 1 ? '' : kind === 'index' ? 'es' : 's'}`,
  };
}

function summarizeG001(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};
  const settingsCount = Object.keys(data).length;

  if (settingsCount === 0) {
    return { status: 'info', message: 'No memory settings found' };
  }

  return {
    status: 'info',
    message: `${settingsCount} memory setting${settingsCount > 1 ? 's' : ''} collected`
  };
}

function summarizeG003(nodeData: any): CheckSummary {
  const data = nodeData?.data || {};

  // G003 has settings and deadlock_stats
  const settings = data.settings || {};
  const deadlockStats = data.deadlock_stats;
  const settingsCount = Object.keys(settings).length;

  if (deadlockStats && typeof deadlockStats.deadlocks === 'number' && deadlockStats.deadlocks > 0) {
    return {
      status: 'warning',
      message: `${deadlockStats.deadlocks} deadlock${deadlockStats.deadlocks > 1 ? 's' : ''} detected`
    };
  }

  if (settingsCount === 0) {
    return { status: 'info', message: 'No timeout/lock settings found' };
  }

  return {
    status: 'info',
    message: `${settingsCount} timeout/lock setting${settingsCount > 1 ? 's' : ''} collected`
  };
}
