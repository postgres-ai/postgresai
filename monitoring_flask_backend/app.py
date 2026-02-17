from flask import Flask, request, jsonify, make_response
from prometheus_api_client import PrometheusConnect
import csv
import io
from datetime import datetime, timezone, timedelta
import logging
import os
import re
import boto3
from requests_aws4auth import AWS4Auth
import psycopg2
import psycopg2.extras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smart_truncate_query(query: str, max_length: int = 40) -> str:
    """
    Smart SQL query truncation for display names.

    Provides more informative truncation than simple character limits by:
    1. Stripping leading comments (/* ... */ and -- ...)
    2. For SELECT queries: showing "SELECT ... FROM <table_names>"
    3. For CTEs: showing "WITH cte1, cte2 SELECT ... FROM ..."
    4. Fallback to simple truncation on any parse error

    Args:
        query: The SQL query text
        max_length: Maximum length for the result (default: 40)

    Returns:
        Truncated query string suitable for display
    """
    if not query:
        return ''

    # Store original for fallback (before try block to ensure it's always bound)
    original_query = query

    try:
        # Step 1: Strip ALL block comments /* ... */ (not just leading ones)
        # This handles inline comments like: SELECT /* comment */ FROM ...
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)

        # Step 2: Strip ALL single-line comments -- ... to end of line
        query = re.sub(r'--[^\n]*', ' ', query)

        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # If already short enough, return it
        if len(query) <= max_length:
            return query

        # Step 3: Parse CTEs (WITH clause)
        cte_names = []
        main_query = query  # The main SELECT statement (after CTEs)
        cte_match = re.match(r'^WITH\s+(RECURSIVE\s+)?', query, re.IGNORECASE)
        if cte_match:
            # Find the main SELECT after CTEs by counting parentheses
            # CTEs are: WITH name AS (...), name2 AS (...) SELECT ...
            after_with = query[cte_match.end():]

            # Extract CTE names from the section before the main SELECT
            cte_name_matches = re.findall(r'(\w+)\s+AS\s*\(', after_with, re.IGNORECASE)
            cte_names = cte_name_matches

            # Find the main SELECT by tracking parentheses depth
            # The main SELECT is at depth 0 after all CTE definitions
            paren_depth = 0
            main_select_pos = None
            i = 0
            while i < len(after_with):
                if after_with[i] == '(':
                    paren_depth += 1
                elif after_with[i] == ')':
                    paren_depth -= 1
                elif paren_depth == 0:
                    # Check if we're at a SELECT keyword at depth 0
                    remaining = after_with[i:].upper()
                    if remaining.startswith('SELECT'):
                        main_select_pos = i
                        break
                i += 1

            if main_select_pos is not None:
                main_query = after_with[main_select_pos:]

        # Step 4: Extract FROM clause tables from the main query
        # Use a more robust approach: find FROM and extract until a SQL keyword
        from_tables = []
        from_match = re.search(r'\bFROM\s+', main_query, re.IGNORECASE)
        if from_match:
            # Get everything after FROM
            after_from = main_query[from_match.end():]
            # Find where the FROM clause ends (at a SQL keyword or end of string)
            end_match = re.search(
                r'\b(WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL|ORDER|GROUP|HAVING|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT|FOR|RETURNING)\b',
                after_from, re.IGNORECASE
            )
            if end_match:
                from_clause = after_from[:end_match.start()].strip()
            else:
                from_clause = after_from.strip()

            # Extract table names (handle aliases, schemas, commas)
            # Split by comma and extract first word of each (the table/alias)
            table_parts = from_clause.split(',')
            for part in table_parts:
                part = part.strip()
                if part:
                    # Get the table name (first word, might include schema.table)
                    table_name_match = re.match(r'^([\w.]+)', part)
                    if table_name_match:
                        table_name = table_name_match.group(1)
                        # Skip subqueries (start with parentheses)
                        if not part.startswith('('):
                            from_tables.append(table_name)

        # Step 5: Determine query type and build display string
        query_upper = query.upper()

        if query_upper.startswith('SELECT') or cte_names:
            # Build the smart truncated version (lowercase to match pg_stat_statements)
            if cte_names and from_tables:
                result = f"with {', '.join(cte_names)} select ... from {', '.join(from_tables)}"
            elif cte_names:
                result = f"with {', '.join(cte_names)} select ..."
            elif from_tables:
                result = f"select ... from {', '.join(from_tables)}"
            else:
                # No FROM clause found, fall back to simple truncation
                result = query[:max_length - 3] + '...'

            # If result is still too long, truncate it
            if len(result) > max_length:
                result = result[:max_length - 3] + '...'

            return result

        elif query_upper.startswith('INSERT'):
            # For INSERT, show the table name (lowercase to match pg_stat_statements)
            insert_match = re.match(r'^INSERT\s+INTO\s+(\S+)', query, re.IGNORECASE)
            if insert_match:
                result = f"insert into {insert_match.group(1)} ..."
                if len(result) > max_length:
                    result = result[:max_length - 3] + '...'
                return result

        elif query_upper.startswith('UPDATE'):
            # For UPDATE, show the table name (lowercase to match pg_stat_statements)
            update_match = re.match(r'^UPDATE\s+(\S+)', query, re.IGNORECASE)
            if update_match:
                result = f"update {update_match.group(1)} ..."
                if len(result) > max_length:
                    result = result[:max_length - 3] + '...'
                return result

        elif query_upper.startswith('DELETE'):
            # For DELETE, show the table name (lowercase to match pg_stat_statements)
            delete_match = re.match(r'^DELETE\s+FROM\s+(\S+)', query, re.IGNORECASE)
            if delete_match:
                result = f"delete from {delete_match.group(1)} ..."
                if len(result) > max_length:
                    result = result[:max_length - 3] + '...'
                return result

        # Fallback: simple truncation
        if len(query) > max_length:
            return query[:max_length - 3] + '...'
        return query

    except Exception:
        # On any error, fall back to simple truncation of original
        if len(original_query) > max_length:
            return original_query[:max_length - 3] + '...'
        return original_query


app = Flask(__name__)

# PostgreSQL sink connection for query text lookups
POSTGRES_SINK_URL = os.environ.get('POSTGRES_SINK_URL', 'postgresql://pgwatch@sink-postgres:5432/measurements')

# Prometheus connection - use environment variable with fallback
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:8428')

# Metric name mapping for cleaner CSV output
METRIC_NAME_MAPPING = {
    'calls': 'calls',
    'exec_time_total': 'exec_time',
    'plan_time_total': 'plan_time',
    'rows': 'rows',
    'shared_bytes_hit_total': 'shared_blks_hit',
    'shared_bytes_read_total': 'shared_blks_read',
    'shared_bytes_dirtied_total': 'shared_blks_dirtied', 
    'shared_bytes_written_total': 'shared_blks_written',
    'block_read_total': 'blk_read_time',
    'block_write_total': 'blk_write_time'
}

def get_prometheus_client():
    """Get Prometheus client connection"""
    try:
        auth = None
        disable_ssl = True

        if os.environ.get('ENABLE_AMP', 'false').lower() == 'true':
            region = os.environ.get('AWS_REGION', 'us-east-1')
            service = 'aps'
            
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials:
                auth = AWS4Auth(
                    region=region,
                    service=service,
                    refreshable_credentials=credentials,
                )
            
            # Enable SSL verification for AMP
            disable_ssl = False

        vm_user = os.environ.get('VM_AUTH_USERNAME')
        vm_pass = os.environ.get('VM_AUTH_PASSWORD')
        if not auth and vm_user and vm_pass:
            auth = (vm_user, vm_pass)

        return PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=disable_ssl, auth=auth)
    except Exception as e:
        logger.error(f"Failed to connect to Prometheus: {e}")
        raise


def get_query_texts_from_sink(db_name: str = None, truncation_mode: str = 'smart') -> dict:
    """
    Fetch queryid-to-query text mappings from the PostgreSQL sink database.

    Args:
        db_name: Optional database name to filter results
        truncation_mode: 'smart' for smart truncation, 'raw' for simple truncation

    Returns:
        Dictionary mapping queryid to query text
    """
    query_texts = {}

    conn = None
    try:
        conn = psycopg2.connect(POSTGRES_SINK_URL)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Skip db_name filter if it's empty, "All", or contains special chars
            use_db_filter = db_name and db_name.lower() not in ('all', '') and not db_name.startswith('$')
            if use_db_filter:
                query = """
                    SELECT DISTINCT ON (data->>'queryid')
                        data->>'queryid' as queryid,
                        data->>'query' as query
                    FROM public.pgss_queryid_queries
                    WHERE
                        dbname = %s
                        AND data->>'queryid' IS NOT NULL
                        AND data->>'query' IS NOT NULL
                    ORDER BY data->>'queryid', time DESC
                """
                cursor.execute(query, (db_name,))
            else:
                query = """
                    SELECT DISTINCT ON (data->>'queryid')
                        data->>'queryid' as queryid,
                        data->>'query' as query
                    FROM public.pgss_queryid_queries
                    WHERE
                        data->>'queryid' IS NOT NULL
                        AND data->>'query' IS NOT NULL
                    ORDER BY data->>'queryid', time DESC
                """
                cursor.execute(query)

            for row in cursor:
                queryid = row['queryid']
                query_text = row['query']
                if queryid:
                    if query_text:
                        if truncation_mode == 'raw':
                            # Raw truncation: normalize whitespace and truncate
                            normalized = ' '.join(query_text.split())
                            query_text = (normalized[:147] + '...') if len(normalized) > 150 else normalized
                        else:
                            # Smart truncation (extracts table names)
                            query_text = smart_truncate_query(query_text, 150)
                    else:
                        query_text = ''
                    query_texts[queryid] = query_text
    except Exception as e:
        logger.warning(f"Failed to fetch query texts from sink database: {e}")
    finally:
        if conn:
            conn.close()

    return query_texts


def read_version_file(filepath, default='unknown'):
    """Read version information from file"""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return default


# Read version info at startup
APP_VERSION = read_version_file('/VERSION')
APP_BUILD_TS = read_version_file('/BUILD_TS')


@app.route('/version', methods=['GET'])
def version():
    """Return application version and build timestamp as array for Grafana Infinity datasource"""
    display = f"PostgresAI v{APP_VERSION} (built: {APP_BUILD_TS})"
    return jsonify([{
        "version": APP_VERSION,
        "build_ts": APP_BUILD_TS,
        "display": display
    }])


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        prom = get_prometheus_client()
        # Simple query to test connection
        prom.get_current_metric_value(metric_name='up')
        return jsonify({"status": "healthy", "prometheus_url": PROMETHEUS_URL})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/pgss_metrics/csv', methods=['GET'])
def get_pgss_metrics_csv():
    """
    Get pg_stat_statements metrics as CSV with time-based difference calculation

    Query parameters:
    - time_start: Start time (ISO format or Unix timestamp)
    - time_end: End time (ISO format or Unix timestamp)
    - cluster_name: Cluster name filter (optional)
    - node_name: Node name filter (optional)
    - db_name: Database name filter (optional)
    - truncation_mode: 'smart' (default) or 'raw' for query text truncation
    """
    try:
        # Get query parameters
        time_start = request.args.get('time_start')
        time_end = request.args.get('time_end')
        cluster_name = request.args.get('cluster_name')
        node_name = request.args.get('node_name')
        truncation_mode = request.args.get('truncation_mode', 'smart')
        db_name = request.args.get('db_name')

        if not time_start or not time_end:
            return jsonify({"error": "time_start and time_end parameters are required"}), 400

        # Parse time parameters
        try:
            # Try parsing as Unix timestamp first
            start_dt = datetime.fromtimestamp(float(time_start), tz=timezone.utc)
        except ValueError:
            # Try parsing as ISO format
            start_dt = datetime.fromisoformat(time_start.replace('Z', '+00:00'))

        try:
            end_dt = datetime.fromtimestamp(float(time_end), tz=timezone.utc)
        except ValueError:
            end_dt = datetime.fromisoformat(time_end.replace('Z', '+00:00'))

        # Connect to Prometheus
        prom = get_prometheus_client()

        # Build the base query for pg_stat_statements metrics
        base_query = 'pgwatch_pg_stat_statements_calls'

        # Add filters if provided
        filters = []
        if cluster_name:
            filters.append(f'cluster="{cluster_name}"')
        if node_name:
            filters.append(f'instance=~".*{node_name}.*"')
        if db_name:
            filters.append(f'datname="{db_name}"')

        if filters:
            base_query += '{' + ','.join(filters) + '}'

        logger.info(f"Querying Prometheus with base query: {base_query}")

        # Get all pg_stat_statements metrics
        all_metrics = [
            'pgwatch_pg_stat_statements_calls',
            'pgwatch_pg_stat_statements_plans_total',
            'pgwatch_pg_stat_statements_exec_time_total',
            'pgwatch_pg_stat_statements_plan_time_total',
            'pgwatch_pg_stat_statements_rows',
            'pgwatch_pg_stat_statements_shared_bytes_hit_total',
            'pgwatch_pg_stat_statements_shared_bytes_read_total',
            'pgwatch_pg_stat_statements_shared_bytes_dirtied_total',
            'pgwatch_pg_stat_statements_shared_bytes_written_total',
            'pgwatch_pg_stat_statements_block_read_total',
            'pgwatch_pg_stat_statements_block_write_total',
            'pgwatch_pg_stat_statements_wal_records',
            'pgwatch_pg_stat_statements_wal_fpi',
            'pgwatch_pg_stat_statements_wal_bytes',
            'pgwatch_pg_stat_statements_temp_bytes_read',
            'pgwatch_pg_stat_statements_temp_bytes_written'
        ]

        # Apply filters to each metric
        filtered_metrics = []
        for metric in all_metrics:
            if filters:
                filtered_metrics.append(f'{metric}{{{",".join(filters)}}}')
            else:
                filtered_metrics.append(metric)

        # Get metrics at start and end times using instant queries
        start_data = []
        end_data = []

        for metric in filtered_metrics:
            try:
                start_metric_data = prom.get_metric_range_data(
                    metric_name=metric,
                    start_time=start_dt - timedelta(minutes=1),
                    end_time=start_dt + timedelta(minutes=1)
                )
                if start_metric_data:
                    start_data.extend(start_metric_data)

                end_metric_data = prom.get_metric_range_data(
                    metric_name=metric,
                    start_time=end_dt - timedelta(minutes=1),
                    end_time=end_dt + timedelta(minutes=1)
                )
                if end_metric_data:
                    end_data.extend(end_metric_data)
            except Exception as e:
                logger.warning(f"Failed to query metric {metric}: {e}")
                continue

        # Fetch query texts from sink database
        # Map legend_label values to truncation mode: displayname_raw_* -> raw, others -> smart
        trunc_mode = 'raw' if 'raw' in truncation_mode.lower() else 'smart'
        query_texts = get_query_texts_from_sink(db_name, truncation_mode=trunc_mode)
        logger.info(f"Fetched {len(query_texts)} query texts from sink database (mode: {trunc_mode})")

        # Process the data to calculate differences
        csv_data = process_pgss_data(start_data, end_data, start_dt, end_dt, query_texts)

        # Create CSV response
        output = io.StringIO()
        if csv_data:
            # Define explicit field order with queryid first, query_text second, then duration, then metrics with their rates
            base_fields = ['queryid', 'query_text', 'duration_seconds']
            all_metric_fields = []

            # Get metric fields from the mapping in specific order with their rates
            desired_order = [
                'calls', 'exec_time', 'plan_time', 'rows', 'shared_blks_hit',
                'shared_blks_read', 'shared_blks_dirtied', 'shared_blks_written',
                'blk_read_time', 'blk_write_time'
            ]

            for display_name in desired_order:
                if display_name in METRIC_NAME_MAPPING.values():
                    all_metric_fields.append(display_name)
                    all_metric_fields.append(f'{display_name}_per_sec')
                    all_metric_fields.append(f'{display_name}_per_call')

            # Combine all fields in desired order
            all_fields = base_fields + all_metric_fields
            
            writer = csv.DictWriter(output, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(csv_data)
        
        csv_content = output.getvalue()
        output.close()

        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=pgss_metrics_{start_dt.strftime("%Y%m%d_%H%M%S")}_{end_dt.strftime("%Y%m%d_%H%M%S")}.csv'

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

def process_pgss_data(start_data, end_data, start_time, end_time, query_texts=None):
    """
    Process pg_stat_statements data and calculate differences between start and end times

    Args:
        start_data: Prometheus data at start time
        end_data: Prometheus data at end time
        start_time: Start datetime
        end_time: End datetime
        query_texts: Optional dictionary mapping queryid to query text
    """
    if query_texts is None:
        query_texts = {}

    # Convert Prometheus data to dictionaries
    start_metrics = prometheus_to_dict(start_data, start_time)
    end_metrics = prometheus_to_dict(end_data, end_time)

    if not start_metrics and not end_metrics:
        return []

    # Create a combined dictionary with all unique query identifiers
    all_keys = set()
    all_keys.update(start_metrics.keys())
    all_keys.update(end_metrics.keys())

    result_rows = []

    # Calculate differences for each query
    for key in all_keys:
        start_metric = start_metrics.get(key, {})
        end_metric = end_metrics.get(key, {})

        # Extract identifier components from key
        db_name, query_id, user, instance = key

        # Calculate actual duration from metric timestamps
        start_timestamp = start_metric.get('timestamp')
        end_timestamp = end_metric.get('timestamp')

        if start_timestamp and end_timestamp:
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            actual_duration = (end_dt - start_dt).total_seconds()
        else:
            # Fallback to query parameter duration if timestamps are missing
            actual_duration = (end_time - start_time).total_seconds()

        # Create result row with query text
        row = {
            'queryid': query_id,
            'query_text': query_texts.get(query_id, ''),
            'duration_seconds': actual_duration
        }

        # Numeric columns to calculate differences for (using original metric names)
        numeric_cols = list(METRIC_NAME_MAPPING.keys())
        
        # Calculate differences and rates
        for col in numeric_cols:
            start_val = start_metric.get(col, 0)
            end_val = end_metric.get(col, 0)
            diff = end_val - start_val
            
            # Use simplified display name for CSV columns
            display_name = METRIC_NAME_MAPPING[col]
            
            # Convert bytes to blocks for block-related metrics (PostgreSQL uses 8KB blocks)
            if 'blks' in display_name and 'bytes' in col:
                diff = diff / 8192  # Convert bytes to 8KB blocks
            
            row[display_name] = diff
            
            # Calculate rates per second
            if row['duration_seconds'] > 0:
                row[f'{display_name}_per_sec'] = diff / row['duration_seconds']
            else:
                row[f'{display_name}_per_sec'] = 0
                
            # Calculate per-call averages
            calls_diff = row.get('calls', 0)
            if calls_diff > 0:
                row[f'{display_name}_per_call'] = diff / calls_diff
            else:
                row[f'{display_name}_per_call'] = 0

        result_rows.append(row)

    # Sort by total execution time difference (descending)
    result_rows.sort(key=lambda x: x.get('exec_time', 0), reverse=True)

    return result_rows

def prometheus_to_dict(prom_data, timestamp):
    """
    Convert Prometheus API response to dictionary keyed by query identifiers
    """
    if not prom_data:
        return {}

    metrics_dict = {}

    for metric_data in prom_data:
        metric = metric_data.get('metric', {})
        values = metric_data.get('values', [])

        if not values:
            continue

        # Get the closest value to our timestamp
        closest_value = min(values, key=lambda x: abs(float(x[0]) - timestamp.timestamp()))

        # Create unique key for this query
        key = (
            metric.get('datname', ''),
            metric.get('queryid', ''),
            metric.get('user', ''),
            metric.get('instance', '')
        )

        # Initialize metric dict if not exists
        if key not in metrics_dict:
            metrics_dict[key] = {
                'timestamp': datetime.fromtimestamp(float(closest_value[0]), tz=timezone.utc).isoformat(),
            }

        # Add metric value
        metric_name = metric.get('__name__', 'pgwatch_pg_stat_statements_calls')
        clean_name = metric_name.replace('pgwatch_pg_stat_statements_', '')

        try:
            metrics_dict[key][clean_name] = float(closest_value[1])
        except (ValueError, IndexError):
            metrics_dict[key][clean_name] = 0

    return metrics_dict

@app.route('/metrics', methods=['GET'])
def list_metrics():
    """List available metrics in Prometheus"""
    try:
        prom = get_prometheus_client()
        metrics = prom.all_metrics()
        pgss_metrics = [m for m in metrics if 'pg_stat_statements' in m]
        return jsonify({"pg_stat_statements_metrics": pgss_metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/metrics', methods=['GET'])
def debug_metrics():
    """
    Debug endpoint to check what metrics are actually available in Prometheus
    """
    try:
        prom = get_prometheus_client()
        
        # Get all available metrics
        all_metrics = prom.all_metrics()
        
        # Filter for pg_btree_bloat metrics
        btree_metrics = [m for m in all_metrics if 'btree_bloat' in m]
        
        # Get sample data for each btree metric
        sample_data = {}
        for metric in btree_metrics[:5]:  # Limit to first 5 to avoid overwhelming
            try:
                result = prom.get_current_metric_value(metric_name=metric)
                sample_data[metric] = {
                    'count': len(result),
                    'sample_labels': [entry.get('metric', {}) for entry in result[:2]]  # First 2 entries
                }
            except Exception as e:
                sample_data[metric] = {'error': str(e)}
        
        return jsonify({
            'all_metrics_count': len(all_metrics),
            'btree_metrics': btree_metrics,
            'sample_data': sample_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/btree_bloat/csv', methods=['GET'])
def get_btree_bloat_csv():
    """
    Get the most recent pg_btree_bloat metrics as a CSV table.
    """
    try:
        # Get query parameters
        cluster_name = request.args.get('cluster_name')
        node_name = request.args.get('node_name')
        db_name = request.args.get('db_name')
        schemaname = request.args.get('schemaname')
        tblname = request.args.get('tblname')
        idxname = request.args.get('idxname')

        # Build label filters
        filters = []
        if cluster_name:
            filters.append(f'cluster="{cluster_name}"')
        if node_name:
            filters.append(f'node_name="{node_name}"')
        if schemaname:
            filters.append(f'schemaname="{schemaname}"')
        if tblname:
            filters.append(f'tblname="{tblname}"')
        if idxname:
            filters.append(f'idxname="{idxname}"')
        if db_name:
            filters.append(f'datname="{db_name}"')

        filter_str = '{' + ','.join(filters) + '}' if filters else ''

        # Metrics to fetch with last_over_time to get only the most recent value
        metric_queries = [
            f'last_over_time(pgwatch_pg_btree_bloat_real_size_mib{filter_str}[1d])',
            f'last_over_time(pgwatch_pg_btree_bloat_extra_size{filter_str}[1d])',
            f'last_over_time(pgwatch_pg_btree_bloat_extra_pct{filter_str}[1d])',
            f'last_over_time(pgwatch_pg_btree_bloat_fillfactor{filter_str}[1d])',
            f'last_over_time(pgwatch_pg_btree_bloat_bloat_size{filter_str}[1d])',
            f'last_over_time(pgwatch_pg_btree_bloat_bloat_pct{filter_str}[1d])',
            f'last_over_time(pgwatch_pg_btree_bloat_is_na{filter_str}[1d])',
        ]

        prom = get_prometheus_client()
        metric_results = {}

        for query in metric_queries:
            try:
                # Use custom_query instead of get_current_metric_value
                result = prom.custom_query(query=query)

                for entry in result:
                    metric_labels = entry.get('metric', {})
                    key = (
                        metric_labels.get('datname', ''),
                        metric_labels.get('schemaname', ''),
                        metric_labels.get('tblname', ''),
                        metric_labels.get('idxname', '')
                    )

                    if key not in metric_results:
                        metric_results[key] = {
                            'database': metric_labels.get('datname', ''),
                            'schemaname': metric_labels.get('schemaname', ''),
                            'tblname': metric_labels.get('tblname', ''),
                            'idxname': metric_labels.get('idxname', ''),
                        }

                    # Extract metric type from query and store value
                    if 'real_size_mib' in query:
                        metric_results[key]['real_size_mib'] = float(entry['value'][1])
                    elif 'extra_size' in query and 'extra_pct' not in query:
                        metric_results[key]['extra_size'] = float(entry['value'][1])
                    elif 'extra_pct' in query:
                        metric_results[key]['extra_pct'] = float(entry['value'][1])
                    elif 'fillfactor' in query:
                        metric_results[key]['fillfactor'] = float(entry['value'][1])
                    elif 'bloat_size' in query:
                        metric_results[key]['bloat_size'] = float(entry['value'][1])
                    elif 'bloat_pct' in query:
                        metric_results[key]['bloat_pct'] = float(entry['value'][1])
                    elif 'is_na' in query:
                        metric_results[key]['is_na'] = int(float(entry['value'][1]))

            except Exception as e:
                logger.warning(f"Failed to query: {query}, error: {e}")
                continue

        # Prepare CSV output
        output = io.StringIO()
        fieldnames = [
            'database', 'schemaname', 'tblname', 'idxname',
            'real_size_mib', 'extra_size', 'extra_pct', 'fillfactor',
            'bloat_size', 'bloat_pct', 'is_na'
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_results.values():
            writer.writerow(row)

        csv_content = output.getvalue()
        output.close()

        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=btree_bloat_latest.csv'
        return response

    except Exception as e:
        logger.error(f"Error processing btree bloat request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/table_info/csv', methods=['GET'])
def get_table_info_csv():
    """
    Get comprehensive table information including size metrics, tuple statistics, and I/O statistics as a CSV table.
    Supports both instant queries (without time parameters) and rate calculations over a time period.
    
    Query parameters:
    - time_start: Start time (ISO format or Unix timestamp) - optional
    - time_end: End time (ISO format or Unix timestamp) - optional
    - cluster_name: Cluster name filter (optional)
    - node_name: Node name filter (optional)
    - db_name: Database name filter (optional)
    - schemaname: Schema name filter (optional, supports regex with ~)
    - tblname: Table name filter (optional)
    """
    try:
        # Get query parameters
        time_start = request.args.get('time_start')
        time_end = request.args.get('time_end')
        cluster_name = request.args.get('cluster_name')
        node_name = request.args.get('node_name')
        db_name = request.args.get('db_name')
        schemaname = request.args.get('schemaname')
        tblname = request.args.get('tblname')

        # Determine if we should calculate rates
        calculate_rates = bool(time_start and time_end)
        
        if calculate_rates:
            # Parse time parameters
            try:
                start_dt = datetime.fromtimestamp(float(time_start), tz=timezone.utc)
            except ValueError:
                start_dt = datetime.fromisoformat(time_start.replace('Z', '+00:00'))

            try:
                end_dt = datetime.fromtimestamp(float(time_end), tz=timezone.utc)
            except ValueError:
                end_dt = datetime.fromisoformat(time_end.replace('Z', '+00:00'))

        # Build label filters
        filters = []
        if cluster_name:
            filters.append(f'cluster="{cluster_name}"')
        if node_name:
            filters.append(f'node_name="{node_name}"')
        if schemaname:
            # Support regex pattern matching with =~
            filters.append(f'schemaname=~"{schemaname}"')
        if tblname:
            filters.append(f'tblname="{tblname}"')
        if db_name:
            filters.append(f'datname="{db_name}"')

        filter_str = '{' + ','.join(filters) + '}' if filters else ''

        prom = get_prometheus_client()
        
        # Define base metrics to query (without last_over_time wrapper for rate calculation)
        base_metrics = {
            # Size metrics
            'total_size': f'pgwatch_pg_class_total_relation_size_bytes{filter_str}',
            'table_size': f'pgwatch_table_size_detailed_table_main_size_b{filter_str}',
            'index_size': f'pgwatch_table_size_detailed_table_indexes_size_b{filter_str}',
            'toast_size': f'pgwatch_table_size_detailed_total_toast_size_b{filter_str}',
            # Scan statistics
            'seq_scan': f'pgwatch_pg_stat_all_tables_seq_scan{filter_str}',
            'idx_scan': f'pgwatch_pg_stat_all_tables_idx_scan{filter_str}',
            # Tuple statistics
            'n_tup_ins': f'pgwatch_table_stats_n_tup_ins{filter_str}',
            'n_tup_upd': f'pgwatch_table_stats_n_tup_upd{filter_str}',
            'n_tup_del': f'pgwatch_table_stats_n_tup_del{filter_str}',
            'n_tup_hot_upd': f'pgwatch_table_stats_n_tup_hot_upd{filter_str}',
            # I/O statistics
            'heap_blks_read': f'pgwatch_pg_statio_all_tables_heap_blks_read{filter_str}',
            'heap_blks_hit': f'pgwatch_pg_statio_all_tables_heap_blks_hit{filter_str}',
            'idx_blks_read': f'pgwatch_pg_statio_all_tables_idx_blks_read{filter_str}',
            'idx_blks_hit': f'pgwatch_pg_statio_all_tables_idx_blks_hit{filter_str}',
        }
        
        if calculate_rates:
            # Get metrics at start and end times
            start_data = {}
            end_data = {}
            
            for metric_name, metric_query in base_metrics.items():
                try:
                    # Get data at start time
                    start_result = prom.get_metric_range_data(
                        metric_name=metric_query,
                        start_time=start_dt - timedelta(minutes=1),
                        end_time=start_dt + timedelta(minutes=1)
                    )
                    if start_result:
                        start_data[metric_name] = start_result
                    
                    # Get data at end time
                    end_result = prom.get_metric_range_data(
                        metric_name=metric_query,
                        start_time=end_dt - timedelta(minutes=1),
                        end_time=end_dt + timedelta(minutes=1)
                    )
                    if end_result:
                        end_data[metric_name] = end_result
                except Exception as e:
                    logger.warning(f"Failed to query metric {metric_name}: {e}")
                    continue
            
            # Process the data to calculate rates
            metric_results = process_table_stats_with_rates(start_data, end_data, start_dt, end_dt)
        else:
            # Get instant values using last_over_time
            metric_results = {}
            for metric_name, metric_query in base_metrics.items():
                try:
                    result = prom.custom_query(query=f'last_over_time({metric_query}[1d])')
                    for entry in result:
                        metric_labels = entry.get('metric', {})
                        
                        # Use different key depending on label names
                        schema_label = metric_labels.get('schemaname') or metric_labels.get('schema', '')
                        table_label = metric_labels.get('relname') or metric_labels.get('table_name') or metric_labels.get('tblname', '')
                        
                        key = (
                            metric_labels.get('datname', ''),
                            schema_label,
                            table_label,
                        )
                        
                        if key not in metric_results:
                            metric_results[key] = {
                                'database': metric_labels.get('datname', ''),
                                'schema': schema_label,
                                'table_name': table_label,
                            }
                        
                        value = float(entry['value'][1])
                        metric_results[key][metric_name] = value
                except Exception as e:
                    logger.warning(f"Failed to query metric {metric_name}: {e}")
                    continue

        # Prepare CSV output
        output = io.StringIO()
        
        if calculate_rates:
            # Fields with rate calculations
            fieldnames = [
                'schema', 'table_name',
                # Size metrics (bytes)
                'total_size', 'table_size', 'index_size', 'toast_size',
                # Scan statistics with rates
                'seq_scans', 'seq_scans_per_sec',
                'idx_scans', 'idx_scans_per_sec',
                # Tuple statistics with rates
                'inserts', 'inserts_per_sec',
                'updates', 'updates_per_sec',
                'deletes', 'deletes_per_sec',
                'hot_updates', 'hot_updates_per_sec',
                # I/O statistics with rates (in bytes using block_size)
                'heap_blks_read', 'heap_blks_read_per_sec',
                'heap_blks_hit', 'heap_blks_hit_per_sec',
                'idx_blks_read', 'idx_blks_read_per_sec',
                'idx_blks_hit', 'idx_blks_hit_per_sec',
                'duration_seconds'
            ]
        else:
            # Fields without rate calculations
            fieldnames = [
                'schema', 'table_name',
                'total_size', 'table_size', 'index_size', 'toast_size',
                'seq_scan', 'idx_scan',
                'n_tup_ins', 'n_tup_upd', 'n_tup_del', 'n_tup_hot_upd',
                'heap_blks_read', 'heap_blks_hit',
                'idx_blks_read', 'idx_blks_hit'
            ]
            
            # Remove 'database' field from rows if present (not in fieldnames)
            for row in metric_results.values():
                row.pop('database', None)
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write rows (handle both dict and list)
        if isinstance(metric_results, dict):
            rows = metric_results.values()
        else:
            rows = metric_results
        
        for row in rows:
            writer.writerow(row)

        csv_content = output.getvalue()
        output.close()

        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        
        if calculate_rates:
            filename = f'table_stats_{start_dt.strftime("%Y%m%d_%H%M%S")}_{end_dt.strftime("%Y%m%d_%H%M%S")}.csv'
        else:
            filename = 'table_stats_latest.csv'
        
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response

    except Exception as e:
        logger.error(f"Error processing table stats request: {e}")
        return jsonify({"error": str(e)}), 500

def process_table_stats_with_rates(start_data, end_data, start_time, end_time):
    """
    Process table statistics and calculate rates between start and end times
    """
    # Convert data to dictionaries
    start_metrics = prometheus_table_to_dict(start_data, start_time)
    end_metrics = prometheus_table_to_dict(end_data, end_time)
    
    if not start_metrics and not end_metrics:
        return []
    
    # Get all unique table identifiers
    all_keys = set()
    all_keys.update(start_metrics.keys())
    all_keys.update(end_metrics.keys())
    
    result_rows = []
    
    for key in all_keys:
        start_metric = start_metrics.get(key, {})
        end_metric = end_metrics.get(key, {})
        
        # Extract identifier components from key
        db_name, schema_name, table_name = key
        
        # Calculate actual duration
        start_timestamp = start_metric.get('timestamp')
        end_timestamp = end_metric.get('timestamp')
        
        if start_timestamp and end_timestamp:
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            actual_duration = (end_dt - start_dt).total_seconds()
        else:
            actual_duration = (end_time - start_time).total_seconds()
        
        # Create result row
        row = {
            'schema': schema_name,
            'table_name': table_name,
            'duration_seconds': actual_duration
        }
        
        # Counter metrics to calculate differences and rates
        counter_metrics = [
            'seq_scan', 'idx_scan', 'n_tup_ins', 'n_tup_upd', 
            'n_tup_del', 'n_tup_hot_upd', 'heap_blks_read', 'heap_blks_hit',
            'idx_blks_read', 'idx_blks_hit'
        ]
        
        # Mapping for display names
        display_names = {
            'seq_scan': 'seq_scans',
            'idx_scan': 'idx_scans',
            'n_tup_ins': 'inserts',
            'n_tup_upd': 'updates',
            'n_tup_del': 'deletes',
            'n_tup_hot_upd': 'hot_updates',
        }
        
        # Calculate differences and rates
        for metric in counter_metrics:
            start_val = start_metric.get(metric, 0)
            end_val = end_metric.get(metric, 0)
            diff = end_val - start_val
            
            # Use display name if available
            display_name = display_names.get(metric, metric)
            
            row[display_name] = diff
            
            # Calculate rate per second
            if actual_duration > 0:
                row[f'{display_name}_per_sec'] = diff / actual_duration
            else:
                row[f'{display_name}_per_sec'] = 0
        
        # Size metrics (just use end values, these don't need rates)
        for size_metric in ['total_size', 'table_size', 'index_size', 'toast_size']:
            row[size_metric] = end_metric.get(size_metric, 0)
        
        result_rows.append(row)
    
    # Sort by total size descending
    result_rows.sort(key=lambda x: x.get('total_size', 0), reverse=True)
    
    return result_rows

def prometheus_table_to_dict(prom_data, timestamp):
    """
    Convert Prometheus table metrics to dictionary keyed by table identifiers
    """
    if not prom_data:
        return {}
    
    metrics_dict = {}
    
    for metric_name, metric_results in prom_data.items():
        for metric_data in metric_results:
            metric = metric_data.get('metric', {})
            values = metric_data.get('values', [])
            
            if not values:
                continue
            
            # Get the closest value to our timestamp
            closest_value = min(values, key=lambda x: abs(float(x[0]) - timestamp.timestamp()))
            
            # Handle different label names
            schema_label = metric.get('schemaname') or metric.get('schema', '')
            table_label = metric.get('relname') or metric.get('table_name') or metric.get('tblname', '')
            
            # Create unique key for this table
            key = (
                metric.get('datname', ''),
                schema_label,
                table_label,
            )
            
            # Initialize metric dict if not exists
            if key not in metrics_dict:
                metrics_dict[key] = {
                    'timestamp': datetime.fromtimestamp(float(closest_value[0]), tz=timezone.utc).isoformat(),
                }
            
            # Add metric value
            try:
                metrics_dict[key][metric_name] = float(closest_value[1])
            except (ValueError, IndexError):
                metrics_dict[key][metric_name] = 0
    
    return metrics_dict


@app.route('/query_texts', methods=['GET'])
def get_query_texts():
    """
    Get queryid-to-query text mappings for use in Grafana TopN chart legends.

    Returns a JSON array of objects with 'queryid', 'query_text', and 'displayName' fields,
    suitable for use with Grafana's Infinity datasource and "Config from query results" transformation.

    The 'displayName' field contains a smart-truncated query string suitable for chart legends:
    - Leading comments (/* ... */ and -- ...) are stripped
    - SELECT queries show: "SELECT ... FROM <table_names>"
    - CTEs show: "WITH cte1, cte2 SELECT ... FROM ..."
    - INSERT/UPDATE/DELETE show the target table name
    - Falls back to simple truncation on parse errors

    Query parameters:
    - db_name: Database name filter (optional)
    - truncate: Max characters for query text (default: 40 for chart legends)
    """
    try:
        db_name = request.args.get('db_name')

        # Validate truncate parameter
        try:
            truncate_len = int(request.args.get('truncate', 40))
            if truncate_len < 1 or truncate_len > 10000:
                return jsonify({'error': 'truncate must be between 1 and 10000'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'truncate must be a valid integer'}), 400

        query_texts = {}

        conn = None
        try:
            conn = psycopg2.connect(POSTGRES_SINK_URL)
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Skip db_name filter if it's empty, "All", or contains special chars
                use_db_filter = db_name and db_name.lower() not in ('all', '') and not db_name.startswith('$')
                if use_db_filter:
                    query = """
                        SELECT DISTINCT ON (data->>'queryid')
                            data->>'queryid' as queryid,
                            data->>'query' as query
                        FROM public.pgss_queryid_queries
                        WHERE
                            dbname = %s
                            AND data->>'queryid' IS NOT NULL
                            AND data->>'query' IS NOT NULL
                        ORDER BY data->>'queryid', time DESC
                    """
                    cursor.execute(query, (db_name,))
                else:
                    query = """
                        SELECT DISTINCT ON (data->>'queryid')
                            data->>'queryid' as queryid,
                            data->>'query' as query
                        FROM public.pgss_queryid_queries
                        WHERE
                            data->>'queryid' IS NOT NULL
                            AND data->>'query' IS NOT NULL
                        ORDER BY data->>'queryid', time DESC
                    """
                    cursor.execute(query)

                for row in cursor:
                    queryid = row['queryid']
                    query_text = row['query']
                    if queryid:
                        # Smart truncation for chart legend display
                        # Use defensive check for None (despite SQL filter, drivers may return None)
                        query_text = smart_truncate_query(query_text or '', truncate_len)
                        query_texts[queryid] = query_text or ''
        except Exception as e:
            logger.warning(f"Failed to fetch query texts from sink database: {e}")
        finally:
            if conn:
                conn.close()

        # Return as JSON array for Grafana Infinity datasource
        # Include 'displayName' for use with "Config from query results" transformation
        # displayName = query_text for legend series name (readable)
        # queryid remains separate for navigation links to Dashboard 3
        result = [
            {
                'queryid': qid,
                'query_text': qtext,
                'displayName': qtext if qtext else qid
            }
            for qid, qtext in query_texts.items()
        ]

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching query texts: {e}")
        return jsonify({"error": str(e)}), 500


def _escape_prometheus_label(value):
    """Escape a string for use as a Prometheus label value.

    Handles:
    - Backslashes: \\ -> \\\\
    - Double quotes: " -> \\"
    - Newlines, carriage returns, tabs: replaced with space
    """
    if not value:
        return ""
    return (value
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', ' ')
            .replace('\r', ' ')
            .replace('\t', ' '))


@app.route('/query_info_metrics', methods=['GET'])
def get_query_info_metrics():
    """
    Return query info metrics in Prometheus exposition format.

    This endpoint is designed to be scraped by VictoriaMetrics/Prometheus
    to create pgwatch_query_info{queryid="...", displayname="...", ...} metrics
    that can be used with group_left() to add display names to chart legends.

    The metrics include multiple truncation levels for dashboard flexibility:
    - displayname: Short version (30 chars) - default for legends
    - displayname_medium: Medium version (60 chars)
    - displayname_long: Long version (100 chars)

    Query parameters:
    - db_name: Database name filter (optional)
    """
    try:
        db_name = request.args.get('db_name')

        # Truncation lengths for different display modes
        TRUNCATE_SHORT = 30
        TRUNCATE_MEDIUM = 60
        TRUNCATE_LONG = 100

        query_data = {}

        conn = None
        try:
            conn = psycopg2.connect(POSTGRES_SINK_URL)
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Skip db_name filter if it's empty, "All", or contains special chars
                use_db_filter = db_name and db_name.lower() not in ('all', '') and not db_name.startswith('$')
                if use_db_filter:
                    query = """
                        SELECT DISTINCT ON (data->>'queryid')
                            data->>'queryid' as queryid,
                            data->>'query' as query
                        FROM public.pgss_queryid_queries
                        WHERE
                            dbname = %s
                            AND data->>'queryid' IS NOT NULL
                            AND data->>'query' IS NOT NULL
                        ORDER BY data->>'queryid', time DESC
                    """
                    cursor.execute(query, (db_name,))
                else:
                    query = """
                        SELECT DISTINCT ON (data->>'queryid')
                            data->>'queryid' as queryid,
                            data->>'query' as query
                        FROM public.pgss_queryid_queries
                        WHERE
                            data->>'queryid' IS NOT NULL
                            AND data->>'query' IS NOT NULL
                        ORDER BY data->>'queryid', time DESC
                    """
                    cursor.execute(query)

                for row in cursor:
                    queryid = row['queryid']
                    query_text = row['query'] or ''  # Defensive check for None
                    if queryid:
                        # Normalize whitespace for raw truncation
                        normalized_text = ' '.join(query_text.split()) if query_text else ''

                        # Smart truncation (extracts table names, CTEs, etc.)
                        short_text = smart_truncate_query(query_text, TRUNCATE_SHORT)
                        medium_text = smart_truncate_query(query_text, TRUNCATE_MEDIUM)
                        long_text = smart_truncate_query(query_text, TRUNCATE_LONG)

                        # Format: "queryid | query text" - queryid prefix for easy copy-paste
                        short = _escape_prometheus_label(f"{queryid} | {short_text}" if short_text else queryid)
                        medium = _escape_prometheus_label(f"{queryid} | {medium_text}" if medium_text else queryid)
                        long = _escape_prometheus_label(f"{queryid} | {long_text}" if long_text else queryid)

                        # Raw truncation (simple character limit, no smart extraction)
                        raw_short_text = (normalized_text[:TRUNCATE_SHORT-3] + '...') if len(normalized_text) > TRUNCATE_SHORT else normalized_text
                        raw_medium_text = (normalized_text[:TRUNCATE_MEDIUM-3] + '...') if len(normalized_text) > TRUNCATE_MEDIUM else normalized_text
                        raw_long_text = (normalized_text[:TRUNCATE_LONG-3] + '...') if len(normalized_text) > TRUNCATE_LONG else normalized_text

                        raw_short = _escape_prometheus_label(f"{queryid} | {raw_short_text}" if raw_short_text else queryid)
                        raw_medium = _escape_prometheus_label(f"{queryid} | {raw_medium_text}" if raw_medium_text else queryid)
                        raw_long = _escape_prometheus_label(f"{queryid} | {raw_long_text}" if raw_long_text else queryid)

                        # Full query text (no truncation) - limit to 500 chars for Prometheus label safety
                        full_text = normalized_text[:500] if normalized_text else ''
                        full = _escape_prometheus_label(f"{queryid} | {full_text}" if full_text else queryid)

                        # Just queryid (for users who prefer numeric IDs)
                        queryid_only = _escape_prometheus_label(queryid)

                        query_data[queryid] = {
                            'short': short,
                            'medium': medium,
                            'long': long,
                            'raw_short': raw_short,
                            'raw_medium': raw_medium,
                            'raw_long': raw_long,
                            'full': full,
                            'queryid_only': queryid_only,
                        }
        except Exception as e:
            logger.warning(f"Failed to fetch query texts from sink database: {e}")
        finally:
            if conn:
                conn.close()

        # Generate Prometheus exposition format
        lines = ['# HELP pgwatch_query_info Query ID to display name mapping for chart legends']
        lines.append('# TYPE pgwatch_query_info gauge')

        for queryid, names in query_data.items():
            # Include all truncation levels as labels
            lines.append(
                f'pgwatch_query_info{{queryid="{queryid}",'
                f'displayname="{names["short"]}",'
                f'displayname_medium="{names["medium"]}",'
                f'displayname_long="{names["long"]}",'
                f'displayname_raw_short="{names["raw_short"]}",'
                f'displayname_raw_medium="{names["raw_medium"]}",'
                f'displayname_raw_long="{names["raw_long"]}",'
                f'displayname_full="{names["full"]}",'
                f'displayname_queryid="{names["queryid_only"]}"}} 1'
            )

        response = make_response('\n'.join(lines) + '\n')
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        return response

    except Exception as e:
        logger.error(f"Error generating query info metrics: {e}")
        return f"# Error: {str(e)}\n", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)