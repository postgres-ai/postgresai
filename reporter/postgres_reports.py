#!/usr/bin/env python3
"""
PostgreSQL Reports Generator using PromQL

This script generates JSON reports containing Observations for specific PostgreSQL
check types (A002, A003, A004, A007, D004, F001, F004, F005, H001, H002, H004,
K001, K003, K004, K005, K006, K007, K008, M001, M002, M003, N001) by querying
Prometheus metrics using PromQL.

IMPORTANT: Scope of this module
-------------------------------
This module ONLY generates JSON reports with raw Observations (data collected
from Prometheus/PostgreSQL). The following are explicitly OUT OF SCOPE:

  - Converting JSON reports to other formats (Markdown, HTML, PDF, etc.)
  - Generating Conclusions based on Observations
  - Generating Recommendations based on Conclusions
  - Any report rendering or presentation logic

These responsibilities are handled by separate components in the system.
The JSON output from this module serves as input for downstream processing.
"""

__version__ = "1.0.2"

import requests
import json
import time
import re
import gc
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Sequence
import argparse
import sys
import os
from pathlib import Path
try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

import boto3
from requests_aws4auth import AWS4Auth

from reporter.logger import logger


class PostgresReportGenerator:
    # Default databases to always exclude
    DEFAULT_EXCLUDED_DATABASES = {'template0', 'template1', 'rdsadmin', 'azure_maintenance', 'cloudsqladmin'}

    # Settings filter lists for reports based on A003
    D004_SETTINGS = [
        'pg_stat_statements.max',
        'pg_stat_statements.track',
        'pg_stat_statements.track_utility',
        'pg_stat_statements.save',
        'pg_stat_statements.track_planning',
        'shared_preload_libraries',
        'track_activities',
        'track_counts',
        'track_functions',
        'track_io_timing',
        'track_wal_io_timing'
    ]

    F001_SETTINGS = [
        'autovacuum',
        'autovacuum_analyze_scale_factor',
        'autovacuum_analyze_threshold',
        'autovacuum_freeze_max_age',
        'autovacuum_max_workers',
        'autovacuum_multixact_freeze_max_age',
        'autovacuum_naptime',
        'autovacuum_vacuum_cost_delay',
        'autovacuum_vacuum_cost_limit',
        'autovacuum_vacuum_insert_scale_factor',
        'autovacuum_vacuum_scale_factor',
        'autovacuum_vacuum_threshold',
        'autovacuum_work_mem',
        'vacuum_cost_delay',
        'vacuum_cost_limit',
        'vacuum_cost_page_dirty',
        'vacuum_cost_page_hit',
        'vacuum_cost_page_miss',
        'vacuum_freeze_min_age',
        'vacuum_freeze_table_age',
        'vacuum_multixact_freeze_min_age',
        'vacuum_multixact_freeze_table_age'
    ]

    G001_SETTINGS = [
        'shared_buffers',
        'work_mem',
        'maintenance_work_mem',
        'effective_cache_size',
        'autovacuum_work_mem',
        'max_wal_size',
        'min_wal_size',
        'wal_buffers',
        'checkpoint_completion_target',
        'max_connections',
        'max_prepared_transactions',
        'max_locks_per_transaction',
        'max_pred_locks_per_transaction',
        'max_pred_locks_per_relation',
        'max_pred_locks_per_page',
        'logical_decoding_work_mem',
        'hash_mem_multiplier',
        'temp_buffers',
        'shared_preload_libraries',
        'dynamic_shared_memory_type',
        'huge_pages',
        'max_files_per_process',
        'max_stack_depth'
    ]

    def __init__(self, prometheus_url: str = "http://sink-prometheus:9090",
                 postgres_sink_url: str = "postgresql://pgwatch@sink-postgres:5432/measurements",
                 excluded_databases: Optional[List[str]] = None,
                 use_current_time: bool = False):
        """
        Initialize the PostgreSQL report generator.

        Args:
            prometheus_url: URL of the Prometheus instance (default: http://sink-prometheus:9090)
            postgres_sink_url: Connection string for the Postgres sink database
                              (default: postgresql://pgwatch@sink-postgres:5432/measurements)
            excluded_databases: Additional databases to exclude from reports
            use_current_time: If True, use current time instead of flooring to hour boundary.
                             Useful for testing with recently collected data.
        """
        self.prometheus_url = prometheus_url
        self.base_url = f"{prometheus_url}/api/v1"
        self.postgres_sink_url = postgres_sink_url
        self.pg_conn = None
        self.use_current_time = use_current_time
        self._build_metadata = self._load_build_metadata()
        # Combine default exclusions with user-provided exclusions
        self.excluded_databases = self.DEFAULT_EXCLUDED_DATABASES.copy()
        if excluded_databases:
            self.excluded_databases.update(excluded_databases)

        # AWS Managed Prometheus Support
        self.auth = None
        if os.environ.get('ENABLE_AMP', 'false').lower() == 'true':
            region = os.environ.get('AWS_REGION', 'us-east-1')
            service = 'aps'
            
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials:
                self.auth = AWS4Auth(
                    region=region,
                    service=service,
                    refreshable_credentials=credentials,
                )

        vm_user = os.environ.get('VM_AUTH_USERNAME')
        vm_pass = os.environ.get('VM_AUTH_PASSWORD')
        if not self.auth and vm_user and vm_pass:
            self.auth = (vm_user, vm_pass)

    def _read_text_file(self, path: str) -> Optional[str]:
        """Read and strip a small text file. Returns None if missing/empty/unreadable."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
            return value or None
        except Exception:
            return None

    def _load_build_metadata(self) -> Dict[str, Optional[str]]:
        """
        Load build metadata from the container filesystem.

        Defaults:
        - VERSION_FILE: /VERSION
        - BUILD_TS_FILE: /BUILD_TS
        Both paths can be overridden for testing via env:
        - PGAI_VERSION_FILE
        - PGAI_BUILD_TS_FILE
        """
        version_path = os.getenv("PGAI_VERSION_FILE", "/VERSION")
        build_ts_path = os.getenv("PGAI_BUILD_TS_FILE", "/BUILD_TS")
        return {
            "version": self._read_text_file(version_path),
            "build_ts": self._read_text_file(build_ts_path),
        }

    def test_connection(self) -> bool:
        """Test connection to Prometheus."""
        try:
            response = requests.get(f"{self.base_url}/status/config", timeout=10, auth=self.auth)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def connect_postgres_sink(self) -> bool:
        """Connect to Postgres sink database."""
        if not self.postgres_sink_url:
            return False
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required for postgres sink access but is not installed")
        
        try:
            self.pg_conn = psycopg2.connect(self.postgres_sink_url)
            return True
        except Exception as e:
            logger.error(f"Postgres sink connection failed: {e}")
            return False

    def close_postgres_sink(self):
        """Close Postgres sink connection."""
        if self.pg_conn:
            self.pg_conn.close()
            self.pg_conn = None

    def get_index_definitions_from_sink(self, db_name: str = None) -> Dict[str, str]:
        """
        Get index definitions from the Postgres sink database.
        
        Args:
            db_name: Optional database name to filter results
        
        Returns:
            Dictionary mapping index names to their definitions
        """
        if not self.pg_conn:
            if not self.connect_postgres_sink():
                return {}
        
        index_definitions = {}
        
        try:
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor, name='index_defs_cursor') as cursor:
                # Use server-side cursor for memory efficiency with large result sets
                # PERFORMANCE NOTE: This query will use a Seq Scan on index_definitions table.
                # This is acceptable because:
                # 1. This method is called VERY rarely (only during report generation)
                # 2. The table size is expected to remain small (< 10000 rows per database)
                # 3. Current latency is well under 1 second for typical workloads
                # 
                # If the table grows significantly larger (>> 10000 rows) or latency exceeds 1s,
                # consider adding a GIN index on the data JSONB column or materialized view.
                if db_name:
                    query = """
                        select distinct on (data->>'indexrelname')
                            data->>'indexrelname' as indexrelname,
                            data->>'index_definition' as index_definition,
                            dbname
                        from public.index_definitions
                        where dbname = %s
                        order by data->>'indexrelname', time desc
                    """
                    cursor.execute(query, (db_name,))
                else:
                    query = """
                        select distinct on (dbname, data->>'indexrelname')
                            data->>'indexrelname' as indexrelname,
                            data->>'index_definition' as index_definition,
                            dbname
                        from public.index_definitions
                        order by dbname, data->>'indexrelname', time desc
                    """
                    cursor.execute(query)
                
                # Use iterator to fetch rows in batches instead of loading all at once
                for row in cursor:
                    if row['indexrelname']:
                        # Include database name in the key to avoid collisions across databases
                        key = f"{row['dbname']}.{row['indexrelname']}" if not db_name else row['indexrelname']
                        index_definitions[key] = row['index_definition']
        
        except Exception as e:
            logger.error(f"Error fetching index definitions from Postgres sink: {e}")
        
        return index_definitions

    def get_queryid_queries_from_sink(self, query_text_limit: int = 655360, db_names: List[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Get queryid-to-query text mappings from the Postgres sink database.

        Args:
            query_text_limit: Maximum number of characters for each query text (default: 655360)
            db_names: Optional list of database names to filter results (default: fetch all)
        
        Returns:
            Dictionary with database names as keys, containing queryid->query mappings
        """
        if not self.pg_conn:
            if not self.connect_postgres_sink():
                return {}
        
        queries_by_db: Dict[str, Dict[str, str]] = {}
        
        try:
            # Use server-side cursor for memory efficiency with large result sets
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor, name='queryid_cursor') as cursor:
                # Query unique queryid-to-query mappings
                # The pgss_queryid_queries table stores deduplicated queryid->query mappings
                if db_names:
                    query = """
                        select distinct on (dbname, data->>'queryid')
                            dbname,
                            data->>'queryid' as queryid,
                            data->>'query' as query
                        from public.pgss_queryid_queries
                        where
                            dbname = ANY(%s)
                            and data->>'queryid' is not null
                            and data->>'query' is not null
                        order by dbname, data->>'queryid', time desc
                    """
                    cursor.execute(query, (db_names,))
                else:
                    query = """
                        select distinct on (dbname, data->>'queryid')
                            dbname,
                            data->>'queryid' as queryid,
                            data->>'query' as query
                        from public.pgss_queryid_queries
                        where
                            data->>'queryid' is not null
                            and data->>'query' is not null
                        order by dbname, data->>'queryid', time desc
                    """
                    cursor.execute(query)
                
                # Use iterator to fetch rows in batches instead of loading all at once
                for row in cursor:
                    db_name = row['dbname']
                    queryid = row['queryid']
                    query_text = row['query']
                    
                    # Skip if queryid is missing
                    if not queryid:
                        continue
                    
                    # Truncate query text if it exceeds the limit
                    if query_text and len(query_text) > query_text_limit:
                        query_text = query_text[:query_text_limit] + '...'
                    
                    # Initialize database dict if needed
                    if db_name not in queries_by_db:
                        queries_by_db[db_name] = {}
                    
                    queries_by_db[db_name][queryid] = query_text or ''
        
        except Exception as e:
            logger.error(f"Error fetching queryid queries from Postgres sink: {e}")
        
        return queries_by_db

    def query_instant(self, query: str) -> Dict[str, Any]:
        """
        Execute an instant PromQL query.
        
        Args:
            query: PromQL query string
            
        Returns:
            Dictionary containing the query results
        """
        params = {'query': query}

        try:
            response = requests.get(f"{self.base_url}/query", params=params, auth=self.auth)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Query failed with status {response.status_code}: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {}

    def _get_postgres_version_info(self, cluster: str, node_name: str) -> Dict[str, str]:
        """
        Fetch and parse Postgres version information from pgwatch settings metrics.

        Notes:
        - This helper is intentionally defensive: it validates the returned setting_name label
          (tests may stub query responses broadly by metric name substring).
        - Uses a single query with a regex on setting_name to reduce roundtrips.
        """
        # Support both label schemas:
        # - newer/expected-by-tests: setting_name/setting_value
        # - older/pgwatch-tagged:    tag_setting_name/tag_setting_value
        queries = [
            (
                "setting_name",
                f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}", node_name="{node_name}", '
                f'setting_name=~"server_version|server_version_num"}}[3h])',
            ),
            (
                "tag_setting_name",
                f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}", node_name="{node_name}", '
                f'tag_setting_name=~"server_version|server_version_num"}}[3h])',
            ),
        ]

        version_str = None
        version_num = None

        for label_name, query in queries:
            result = self.query_instant(query)
            if result.get("status") != "success":
                continue
            for item in (result.get("data", {}) or {}).get("result", []) or []:
                metric = item.get("metric", {}) or {}
                setting_name = metric.get("setting_name") or metric.get("tag_setting_name") or ""
                setting_value = metric.get("setting_value") or metric.get("tag_setting_value") or ""

                if setting_name == "server_version" and setting_value and not version_str:
                    version_str = setting_value
                elif setting_name == "server_version_num" and setting_value and not version_num:
                    version_num = setting_value

            if version_str or version_num:
                break

        if not (version_str or version_num):
            logger.warning(f"No version data found (cluster={cluster}, node_name={node_name})")

        server_version = version_str or "Unknown"
        version_info: Dict[str, str] = {
            "version": server_version,
            "server_version_num": version_num or "Unknown",
            "server_major_ver": "Unknown",
            "server_minor_ver": "Unknown",
        }

        if server_version != "Unknown":
            # Handle both formats:
            # - "15.3"
            # - "15.3 (Ubuntu 15.3-1.pgdg20.04+1)"
            version_parts = server_version.split()[0].split(".")
            if len(version_parts) >= 1 and version_parts[0]:
                version_info["server_major_ver"] = version_parts[0]
                if len(version_parts) >= 2:
                    version_info["server_minor_ver"] = ".".join(version_parts[1:])
                else:
                    version_info["server_minor_ver"] = "0"

        return version_info

    def generate_a002_version_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate A002 Version Information report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing version information
        """
        logger.info(f"Generating A002 Version Information report for cluster='{cluster}', node_name='{node_name}'...")
        version_info = self._get_postgres_version_info(cluster, node_name)
        return self.format_report_data("A002", {"version": version_info}, node_name)

    def generate_a003_settings_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate A003 PostgreSQL Settings report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing settings information
        """
        logger.info("Generating A003 PostgreSQL Settings report...")

        # Query all PostgreSQL settings using the pgwatch_settings_configured metric with last_over_time
        # This metric has labels for each setting name
        settings_query = f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        result = self.query_instant(settings_query)

        settings_data = {}
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                # Extract setting name from labels
                setting_name = item['metric'].get('setting_name', '')
                setting_value = item['metric'].get('setting_value', '')
                
                # Skip if we don't have a setting name
                if not setting_name:
                    continue

                # Get additional metadata from labels
                category = item['metric'].get('category', 'Other')
                unit = item['metric'].get('unit', '')
                context = item['metric'].get('context', '')
                vartype = item['metric'].get('vartype', '')

                settings_data[setting_name] = {
                    "setting": setting_value,
                    "unit": unit,
                    "category": category,
                    "context": context,
                    "vartype": vartype,
                    "pretty_value": self.format_setting_value(setting_name, setting_value, unit)
                }
        else:
            logger.warning(f"A003 - No settings data returned for cluster={cluster}, node_name={node_name}")
            logger.info(f"Query result status: {result.get('status')}")
            logger.info(f"Query result data: {result.get('data', {})}")

        return self.format_report_data("A003", settings_data, node_name, postgres_version=self._get_postgres_version_info(cluster, node_name))

    def generate_a004_cluster_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate A004 Cluster Information report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing cluster information
        """
        logger.info("Generating A004 Cluster Information report...")

        # Query cluster information
        cluster_queries = {
            'active_connections': f'sum(last_over_time(pgwatch_pg_stat_activity_count{{cluster="{cluster}", node_name="{node_name}", state="active"}}[3h]))',
            'idle_connections': f'sum(last_over_time(pgwatch_pg_stat_activity_count{{cluster="{cluster}", node_name="{node_name}", state="idle"}}[3h]))',
            'total_connections': f'sum(last_over_time(pgwatch_pg_stat_activity_count{{cluster="{cluster}", node_name="{node_name}"}}[3h]))',
            'database_sizes': f'sum(last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h]))',
            'cache_hit_ratio': f'sum(last_over_time(pgwatch_db_stats_blks_hit{{cluster="{cluster}", node_name="{node_name}"}}[3h])) / clamp_min(sum(last_over_time(pgwatch_db_stats_blks_hit{{cluster="{cluster}", node_name="{node_name}"}}[3h])) + sum(last_over_time(pgwatch_db_stats_blks_read{{cluster="{cluster}", node_name="{node_name}"}}[3h])), 1) * 100',
            'transactions_per_sec': f'sum(rate(pgwatch_db_stats_xact_commit{{cluster="{cluster}", node_name="{node_name}"}}[5m])) + sum(rate(pgwatch_db_stats_xact_rollback{{cluster="{cluster}", node_name="{node_name}"}}[5m]))',
            'checkpoints_per_sec': f'sum(rate(pgwatch_pg_stat_bgwriter_checkpoints_timed{{cluster="{cluster}", node_name="{node_name}"}}[5m])) + sum(rate(pgwatch_pg_stat_bgwriter_checkpoints_req{{cluster="{cluster}", node_name="{node_name}"}}[5m]))',
            'deadlocks': f'sum(last_over_time(pgwatch_db_stats_deadlocks{{cluster="{cluster}", node_name="{node_name}"}}[3h]))',
            'temp_files': f'sum(last_over_time(pgwatch_db_stats_temp_files{{cluster="{cluster}", node_name="{node_name}"}}[3h]))',
            'temp_bytes': f'sum(last_over_time(pgwatch_db_stats_temp_bytes{{cluster="{cluster}", node_name="{node_name}"}}[3h]))',
        }

        cluster_data = {}
        for metric_name, query in cluster_queries.items():
            result = self.query_instant(query)
            if result.get('status') == 'success' and result.get('data', {}).get('result'):
                values = result['data']['result']
                if values:
                    latest_value = values[0].get('value', [None, None])[1]
                    cluster_data[metric_name] = {
                        "value": latest_value,
                        "unit": self.get_cluster_metric_unit(metric_name),
                        "description": self.get_cluster_metric_description(metric_name)
                    }

        # Get database sizes
        db_sizes_query = f'last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        db_sizes_result = self.query_instant(db_sizes_query)
        database_sizes = {}

        if db_sizes_result.get('status') == 'success' and db_sizes_result.get('data', {}).get('result'):
            for result in db_sizes_result['data']['result']:
                db_name = result['metric'].get('datname', 'unknown')
                size_bytes = float(result['value'][1])
                database_sizes[db_name] = size_bytes

        return self.format_report_data(
            "A004",
            {
                "general_info": cluster_data,
                "database_sizes": database_sizes,
            },
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_a007_altered_settings_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[
        str, Any]:
        """
        Generate A007 Altered Settings report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing altered settings information
        """
        logger.info("Generating A007 Altered Settings report...")

        # Query settings by source using the pgwatch_settings_is_default metric with last_over_time
        # This returns settings where is_default = 0 (i.e., non-default/altered settings)
        settings_by_source_query = f'last_over_time(pgwatch_settings_is_default{{cluster="{cluster}", node_name="{node_name}"}}[3h]) < 1'
        result = self.query_instant(settings_by_source_query)

        altered_settings = {}
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                # Extract setting information from labels
                setting_name = item['metric'].get('setting_name', '')
                value = item['metric'].get('setting_value', '')
                unit = item['metric'].get('unit', '')
                category = item['metric'].get('category', 'Other')
                
                # Skip if we don't have a setting name
                if not setting_name:
                    continue
                
                pretty_value = self.format_setting_value(setting_name, value, unit)
                altered_settings[setting_name] = {
                    "value": value,
                    "unit": unit,
                    "category": category,
                    "pretty_value": pretty_value
                }
        else:
            logger.warning(f"A007 - No altered settings data returned for cluster={cluster}, node_name={node_name}")
            logger.info(f"Query result status: {result.get('status')}")

        return self.format_report_data("A007", altered_settings, node_name, postgres_version=self._get_postgres_version_info(cluster, node_name))

    # ==================================================================================
    # H001: Invalid Indexes - Observation Data for Decision Tree
    # ==================================================================================
    #
    # This report collects observation data that enables decision tree analysis:
    #   - has_valid_duplicate / valid_duplicate_name: Is there a valid index on same column(s)?
    #   - is_pk / is_unique / constraint_name: Does it back a constraint (UNIQUE / PK)?
    #   - table_row_estimate: Is the table small (< 10K rows)?
    #
    # The JSON report contains ONLY observations (raw data).
    # Recommendations (DROP, RECREATE, UNCERTAIN) are computed at render time:
    #   - CLI: cli/lib/checkup.ts -> getInvalidIndexRecommendation()
    #   - UI: Grafana dashboard or web template applies the same logic
    #
    # ==================================================================================

    def generate_h001_invalid_indexes_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[
        str, Any]:
        """
        Generate H001 Invalid Indexes report with observation data for decision tree.

        Args:
            cluster: Cluster name
            node_name: Node name

        Returns:
            Dictionary containing invalid indexes observation data for decision tree analysis
        """
        logger.info("Generating H001 Invalid Indexes report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)

        # Get database sizes
        db_sizes_query = f'last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        db_sizes_result = self.query_instant(db_sizes_query)
        database_sizes = {}

        if db_sizes_result.get('status') == 'success' and db_sizes_result.get('data', {}).get('result'):
            for result in db_sizes_result['data']['result']:
                db_name = result['metric'].get('datname', 'unknown')
                size_bytes = float(result['value'][1])
                database_sizes[db_name] = size_bytes

        invalid_indexes_by_db = {}
        for db_name in databases:
            # Fetch index definitions from the sink for this database (used to aid remediation)
            index_definitions = self.get_index_definitions_from_sink(db_name)

            # Query all invalid indexes metrics and merge by index key
            # Each field is a separate metric in pgwatch prometheus export
            base_filter = f'cluster="{cluster}", node_name="{node_name}", datname="{db_name}"'

            # Query primary metric (index_size_bytes) - this determines which indexes exist
            size_query = f'last_over_time(pgwatch_pg_invalid_indexes_index_size_bytes{{{base_filter}}}[3h])'
            size_result = self.query_instant(size_query)

            # Build index data keyed by (schema_name, table_name, index_name)
            indexes_data: Dict[tuple, Dict[str, Any]] = {}

            if size_result.get('status') == 'success' and size_result.get('data', {}).get('result'):
                for item in size_result['data']['result']:
                    schema_name = item['metric'].get('schema_name', 'unknown')
                    table_name = item['metric'].get('table_name', 'unknown')
                    index_name = item['metric'].get('index_name', 'unknown')
                    key = (schema_name, table_name, index_name)

                    indexes_data[key] = {
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "index_name": index_name,
                        "relation_name": item['metric'].get('relation_name', f"{schema_name}.{table_name}"),
                        "index_size_bytes": float(item['value'][1]) if item.get('value') else 0,
                        "index_definition": index_definitions.get(index_name, "Definition not available"),
                        "valid_duplicate_name": item['metric'].get('valid_index_name') or None,
                        "valid_duplicate_definition": item['metric'].get('valid_index_definition') or None,
                        "constraint_name": item['metric'].get('constraint_name') or None,
                        # Defaults for boolean/numeric fields (will be updated from separate metrics)
                        "supports_fk": False,
                        "is_pk": False,
                        "is_unique": False,
                        "has_valid_duplicate": False,
                        "table_row_estimate": 0,
                    }

            # Query additional metrics and merge values
            additional_metrics = [
                ('pgwatch_pg_invalid_indexes_supports_fk', 'supports_fk', lambda v: int(float(v)) == 1),
                ('pgwatch_pg_invalid_indexes_is_pk', 'is_pk', lambda v: int(float(v)) == 1),
                ('pgwatch_pg_invalid_indexes_is_unique', 'is_unique', lambda v: int(float(v)) == 1),
                ('pgwatch_pg_invalid_indexes_has_valid_duplicate', 'has_valid_duplicate', lambda v: int(float(v)) == 1),
                ('pgwatch_pg_invalid_indexes_table_row_estimate', 'table_row_estimate', lambda v: int(float(v))),
            ]

            for metric_name, field_name, converter in additional_metrics:
                query = f'last_over_time({metric_name}{{{base_filter}}}[3h])'
                result = self.query_instant(query)
                if result.get('status') == 'success' and result.get('data', {}).get('result'):
                    for item in result['data']['result']:
                        key = (
                            item['metric'].get('schema_name', 'unknown'),
                            item['metric'].get('table_name', 'unknown'),
                            item['metric'].get('index_name', 'unknown'),
                        )
                        if key in indexes_data and item.get('value'):
                            try:
                                indexes_data[key][field_name] = converter(item['value'][1])
                            except (ValueError, TypeError):
                                pass  # Keep default value

            # Convert to list and calculate totals
            invalid_indexes = []
            total_size = 0
            for data in indexes_data.values():
                data["index_size_pretty"] = self.format_bytes(data["index_size_bytes"])
                invalid_indexes.append(data)
                total_size += data["index_size_bytes"]

            # Skip databases with no invalid indexes
            if not invalid_indexes:
                continue

            db_size_bytes = database_sizes.get(db_name, 0)
            invalid_indexes_by_db[db_name] = {
                "invalid_indexes": invalid_indexes,
                "total_count": len(invalid_indexes),
                "total_size_bytes": total_size,
                "total_size_pretty": self.format_bytes(total_size),
                "database_size_bytes": db_size_bytes,
                "database_size_pretty": self.format_bytes(db_size_bytes)
            }

        return self.format_report_data(
            "H001",
            invalid_indexes_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_h002_unused_indexes_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate H002 Unused Indexes report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing unused indexes information
        """
        logger.info("Generating H002 Unused Indexes report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)

        # Get database sizes
        db_sizes_query = f'last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        db_sizes_result = self.query_instant(db_sizes_query)
        database_sizes = {}

        if db_sizes_result.get('status') == 'success' and db_sizes_result.get('data', {}).get('result'):
            for result in db_sizes_result['data']['result']:
                db_name = result['metric'].get('datname', 'unknown')
                size_bytes = float(result['value'][1])
                database_sizes[db_name] = size_bytes

        # Query postmaster uptime to get startup time
        postmaster_uptime_query = f'last_over_time(pgwatch_db_stats_postmaster_uptime_s{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        postmaster_uptime_result = self.query_instant(postmaster_uptime_query)
        
        postmaster_startup_time = None
        postmaster_startup_epoch = None
        if postmaster_uptime_result.get('status') == 'success' and postmaster_uptime_result.get('data', {}).get('result'):
            uptime_seconds = float(postmaster_uptime_result['data']['result'][0]['value'][1]) if postmaster_uptime_result['data']['result'] else None
            if uptime_seconds:
                postmaster_startup_epoch = datetime.now().timestamp() - uptime_seconds
                postmaster_startup_time = datetime.fromtimestamp(postmaster_startup_epoch).isoformat()

        unused_indexes_by_db = {}
        for db_name in databases:
            # Get index definitions from Postgres sink database for this specific database
            index_definitions = self.get_index_definitions_from_sink(db_name)
            # Query stats_reset timestamp for this database
            stats_reset_query = f'last_over_time(pgwatch_stats_reset_stats_reset_epoch{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])'
            stats_reset_result = self.query_instant(stats_reset_query)
            
            stats_reset_epoch = None
            days_since_reset = None
            stats_reset_time = None
            
            if stats_reset_result.get('status') == 'success' and stats_reset_result.get('data', {}).get('result'):
                stats_reset_epoch = float(stats_reset_result['data']['result'][0]['value'][1]) if stats_reset_result['data']['result'] else None
                if stats_reset_epoch:
                    stats_reset_time = datetime.fromtimestamp(stats_reset_epoch).isoformat()
                    days_since_reset = (datetime.now() - datetime.fromtimestamp(stats_reset_epoch)).days

            # Query unused indexes for each database using last_over_time to get most recent value
            unused_indexes_query = f'last_over_time(pgwatch_unused_indexes_index_size_bytes{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])'
            unused_result = self.query_instant(unused_indexes_query)

            unused_indexes = []
            if unused_result.get('status') == 'success' and unused_result.get('data', {}).get('result'):
                for item in unused_result['data']['result']:
                    schema_name = item['metric'].get('schema_name', 'unknown')
                    table_name = item['metric'].get('table_name', 'unknown')
                    index_name = item['metric'].get('index_name', 'unknown')
                    reason = item['metric'].get('reason', 'Unknown')

                    # Get the index size from the metric value
                    index_size_bytes = float(item['value'][1]) if item.get('value') else 0

                    # Query other related metrics for this index
                    idx_scan_query = f'last_over_time(pgwatch_unused_indexes_idx_scan{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}", schema_name="{schema_name}", table_name="{table_name}", index_name="{index_name}"}}[3h])'
                    idx_scan_result = self.query_instant(idx_scan_query)
                    idx_scan = float(idx_scan_result['data']['result'][0]['value'][1]) if idx_scan_result.get('data',
                                                                                                              {}).get(
                        'result') else 0

                    # Get index definition from collected metrics
                    index_definition = index_definitions.get(index_name, "Definition not available")

                    index_data = {
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "index_name": index_name,
                        "index_definition": index_definition,
                        "reason": reason,
                        "idx_scan": idx_scan,
                        "index_size_bytes": index_size_bytes,
                        "idx_is_btree": item['metric'].get('idx_is_btree', 'false') == 'true',
                        "supports_fk": bool(int(item['metric'].get('supports_fk', 0)))
                    }

                    index_data['index_size_pretty'] = self.format_bytes(index_data['index_size_bytes'])

                    unused_indexes.append(index_data)

            # Sort by index size descending
            unused_indexes.sort(key=lambda x: x['index_size_bytes'], reverse=True)
            
            # Skip databases with no unused indexes
            if not unused_indexes:
                continue
            
            total_unused_size = sum(idx['index_size_bytes'] for idx in unused_indexes)

            db_size_bytes = database_sizes.get(db_name, 0)
            unused_indexes_by_db[db_name] = {
                "unused_indexes": unused_indexes,
                "total_count": len(unused_indexes),
                "total_size_bytes": total_unused_size,
                "total_size_pretty": self.format_bytes(total_unused_size),
                "database_size_bytes": db_size_bytes,
                "database_size_pretty": self.format_bytes(db_size_bytes),
                "stats_reset": {
                    "stats_reset_epoch": stats_reset_epoch,
                    "stats_reset_time": stats_reset_time,
                    "days_since_reset": days_since_reset,
                    "postmaster_startup_epoch": postmaster_startup_epoch,
                    "postmaster_startup_time": postmaster_startup_time
                }
            }

        return self.format_report_data(
            "H002",
            unused_indexes_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_h004_redundant_indexes_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[
        str, Any]:
        """
        Generate H004 Redundant Indexes report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing redundant indexes information
        """
        logger.info("Generating H004 Redundant Indexes report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)

        # Get database sizes
        db_sizes_query = f'last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        db_sizes_result = self.query_instant(db_sizes_query)
        database_sizes = {}

        if db_sizes_result.get('status') == 'success' and db_sizes_result.get('data', {}).get('result'):
            for result in db_sizes_result['data']['result']:
                db_name = result['metric'].get('datname', 'unknown')
                size_bytes = float(result['value'][1])
                database_sizes[db_name] = size_bytes

        redundant_indexes_by_db = {}
        for db_name in databases:
            # Fetch index definitions from the sink for this database (used to aid remediation)
            index_definitions = self.get_index_definitions_from_sink(db_name)
            # Query redundant indexes for each database using last_over_time to get most recent value
            redundant_indexes_query = f'last_over_time(pgwatch_redundant_indexes_index_size_bytes{{cluster="{cluster}", node_name="{node_name}", dbname="{db_name}"}}[3h])'
            result = self.query_instant(redundant_indexes_query)

            redundant_indexes = []
            total_size = 0

            if result.get('status') == 'success' and result.get('data', {}).get('result'):
                for item in result['data']['result']:
                    schema_name = item['metric'].get('schema_name', 'unknown')
                    table_name = item['metric'].get('table_name', 'unknown')
                    index_name = item['metric'].get('index_name', 'unknown')
                    relation_name = item['metric'].get('relation_name', f"{schema_name}.{table_name}")
                    access_method = item['metric'].get('access_method', 'unknown')
                    reason = item['metric'].get('reason', 'Unknown')

                    # Get the index size from the metric value
                    index_size_bytes = float(item['value'][1]) if item.get('value') else 0

                    # Query other related metrics for this index
                    table_size_query = f'last_over_time(pgwatch_redundant_indexes_table_size_bytes{{cluster="{cluster}", node_name="{node_name}", dbname="{db_name}", schema_name="{schema_name}", table_name="{table_name}", index_name="{index_name}"}}[3h])'
                    table_size_result = self.query_instant(table_size_query)
                    table_size_bytes = float(
                        table_size_result['data']['result'][0]['value'][1]) if table_size_result.get('data', {}).get(
                        'result') else 0

                    index_usage_query = f'last_over_time(pgwatch_redundant_indexes_index_usage{{cluster="{cluster}", node_name="{node_name}", dbname="{db_name}", schema_name="{schema_name}", table_name="{table_name}", index_name="{index_name}"}}[3h])'
                    index_usage_result = self.query_instant(index_usage_query)
                    index_usage = float(index_usage_result['data']['result'][0]['value'][1]) if index_usage_result.get(
                        'data', {}).get('result') else 0

                    supports_fk_query = f'last_over_time(pgwatch_redundant_indexes_supports_fk{{cluster="{cluster}", node_name="{node_name}", dbname="{db_name}", schema_name="{schema_name}", table_name="{table_name}", index_name="{index_name}"}}[3h])'
                    supports_fk_result = self.query_instant(supports_fk_query)
                    supports_fk = bool(
                        int(supports_fk_result['data']['result'][0]['value'][1])) if supports_fk_result.get('data',
                                                                                                            {}).get(
                        'result') else False

                    # Build redundant_to array from the reason field
                    # The reason field contains comma-separated index names
                    # (the indexes that make this index redundant)
                    # Note: In full mode, index sizes for redundant_to are not available
                    # (would require additional Prometheus queries). Use express mode for sizes.
                    redundant_to = []
                    for idx_name in [r.strip() for r in reason.split(',') if r.strip()]:
                        redundant_to.append({
                            "index_name": idx_name,
                            "index_definition": index_definitions.get(idx_name, "Definition not available"),
                            "index_size_bytes": 0,
                            "index_size_pretty": "N/A"
                        })

                    redundant_index = {
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "index_name": index_name,
                        "relation_name": relation_name,
                        "access_method": access_method,
                        "reason": reason,
                        "index_size_bytes": index_size_bytes,
                        "table_size_bytes": table_size_bytes,
                        "index_usage": index_usage,
                        "supports_fk": supports_fk,
                        "index_definition": index_definitions.get(index_name, "Definition not available"),
                        "index_size_pretty": self.format_bytes(index_size_bytes),
                        "table_size_pretty": self.format_bytes(table_size_bytes),
                        "redundant_to": redundant_to
                    }

                    redundant_indexes.append(redundant_index)
                    total_size += index_size_bytes

            # Sort by index size descending
            redundant_indexes.sort(key=lambda x: x['index_size_bytes'], reverse=True)
            
            # Skip databases with no redundant indexes
            if not redundant_indexes:
                continue

            db_size_bytes = database_sizes.get(db_name, 0)
            redundant_indexes_by_db[db_name] = {
                "redundant_indexes": redundant_indexes,
                "total_count": len(redundant_indexes),
                "total_size_bytes": total_size,
                "total_size_pretty": self.format_bytes(total_size),
                "database_size_bytes": db_size_bytes,
                "database_size_pretty": self.format_bytes(db_size_bytes)
            }

        return self.format_report_data(
            "H004",
            redundant_indexes_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_d004_pgstat_settings_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[
        str, Any]:
        """
        Generate D004 pgstatstatements and pgstatkcache Settings report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing pg_stat_statements and pg_stat_kcache settings information
        """
        logger.info("Generating D004 pgstatstatements and pgstatkcache Settings report...")

        # Define relevant pg_stat_statements and pg_stat_kcache settings
        pgstat_settings = [
            'pg_stat_statements.max',
            'pg_stat_statements.track',
            'pg_stat_statements.track_utility',
            'pg_stat_statements.save',
            'pg_stat_statements.track_planning',
            'shared_preload_libraries',
            'track_activities',
            'track_counts',
            'track_functions',
            'track_io_timing',
            'track_wal_io_timing'
        ]

        # Query all PostgreSQL settings for pg_stat_statements and related using last_over_time
        settings_query = f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        result = self.query_instant(settings_query)

        pgstat_data = {}
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                setting_name = item['metric'].get('setting_name', '')
                
                # Skip if no setting name
                if not setting_name:
                    continue

                # Filter for pg_stat_statements and related settings
                if setting_name in pgstat_settings:
                    setting_value = item['metric'].get('setting_value', '')
                    category = item['metric'].get('category', 'Statistics')
                    unit = item['metric'].get('unit', '')
                    context = item['metric'].get('context', '')
                    vartype = item['metric'].get('vartype', '')

                    pgstat_data[setting_name] = {
                        "setting": setting_value,
                        "unit": unit,
                        "category": category,
                        "context": context,
                        "vartype": vartype,
                        "pretty_value": self.format_setting_value(setting_name, setting_value, unit)
                    }
        else:
            logger.warning(f"D004 - No settings data returned for cluster={cluster}, node_name={node_name}")

        # Check if pg_stat_kcache extension is available and working by querying its metrics
        kcache_status = self._check_pg_stat_kcache_status(cluster, node_name)

        # Check if pg_stat_statements is available and working by querying its metrics  
        pgss_status = self._check_pg_stat_statements_status(cluster, node_name)

        return self.format_report_data(
            "D004",
            {
                "settings": pgstat_data,
                "pg_stat_statements_status": pgss_status,
                "pg_stat_kcache_status": kcache_status,
            },
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def _check_pg_stat_kcache_status(self, cluster: str, node_name: str) -> Dict[str, Any]:
        """
        Check if pg_stat_kcache extension is working by querying its metrics.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing pg_stat_kcache status information
        """
        kcache_queries = {
            'exec_user_time': f'last_over_time(pgwatch_pg_stat_kcache_exec_user_time{{cluster="{cluster}", node_name="{node_name}"}}[3h])',
            'exec_system_time': f'last_over_time(pgwatch_pg_stat_kcache_exec_system_time{{cluster="{cluster}", node_name="{node_name}"}}[3h])',
            'exec_total_time': f'last_over_time(pgwatch_pg_stat_kcache_exec_total_time{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        }

        kcache_status = {
            "extension_available": False,
            "metrics_count": 0,
            "total_exec_time": 0,
            "total_user_time": 0,
            "total_system_time": 0,
            "sample_queries": []
        }

        for metric_name, query in kcache_queries.items():
            result = self.query_instant(query)
            if result.get('status') == 'success' and result.get('data', {}).get('result'):
                kcache_status["extension_available"] = True
                results = result['data']['result']

                for item in results[:5]:  # Get sample of top 5 queries
                    queryid = item['metric'].get('queryid', 'unknown')
                    user = item['metric'].get('tag_user', 'unknown')
                    value = float(item['value'][1]) if item.get('value') else 0

                    # Add to totals
                    if metric_name == 'exec_total_time':
                        kcache_status["total_exec_time"] += value
                        kcache_status["metrics_count"] = len(results)

                        # Store sample query info
                        if len(kcache_status["sample_queries"]) < 5:
                            kcache_status["sample_queries"].append({
                                "queryid": queryid,
                                "user": user,
                                "exec_total_time": value
                            })
                    elif metric_name == 'exec_user_time':
                        kcache_status["total_user_time"] += value
                    elif metric_name == 'exec_system_time':
                        kcache_status["total_system_time"] += value

        return kcache_status

    def _check_pg_stat_statements_status(self, cluster: str, node_name: str) -> Dict[str, Any]:
        """
        Check if pg_stat_statements extension is working by querying its metrics.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing pg_stat_statements status information
        """
        pgss_query = f'last_over_time(pgwatch_pg_stat_statements_calls{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        result = self.query_instant(pgss_query)

        pgss_status = {
            "extension_available": False,
            "metrics_count": 0,
            "total_calls": 0,
            "sample_queries": []
        }

        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            pgss_status["extension_available"] = True
            results = result['data']['result']
            pgss_status["metrics_count"] = len(results)

            for item in results[:5]:  # Get sample of top 5 queries
                queryid = item['metric'].get('queryid', 'unknown')
                user = item['metric'].get('tag_user', 'unknown')
                datname = item['metric'].get('datname', 'unknown')
                calls = float(item['value'][1]) if item.get('value') else 0

                pgss_status["total_calls"] += calls

                # Store sample query info
                if len(pgss_status["sample_queries"]) < 5:
                    pgss_status["sample_queries"].append({
                        "queryid": queryid,
                        "user": user,
                        "database": datname,
                        "calls": calls
                    })

        return pgss_status

    def generate_f001_autovacuum_settings_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[
        str, Any]:
        """
        Generate F001 Autovacuum: Current Settings report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing autovacuum settings information
        """
        logger.info("Generating F001 Autovacuum: Current Settings report...")

        # Define autovacuum related settings
        autovacuum_settings = [
            'autovacuum',
            'autovacuum_analyze_scale_factor',
            'autovacuum_analyze_threshold',
            'autovacuum_freeze_max_age',
            'autovacuum_max_workers',
            'autovacuum_multixact_freeze_max_age',
            'autovacuum_naptime',
            'autovacuum_vacuum_cost_delay',
            'autovacuum_vacuum_cost_limit',
            'autovacuum_vacuum_insert_scale_factor',
            'autovacuum_vacuum_scale_factor',
            'autovacuum_vacuum_threshold',
            'autovacuum_work_mem',
            'vacuum_cost_delay',
            'vacuum_cost_limit',
            'vacuum_cost_page_dirty',
            'vacuum_cost_page_hit',
            'vacuum_cost_page_miss',
            'vacuum_freeze_min_age',
            'vacuum_freeze_table_age',
            'vacuum_multixact_freeze_min_age',
            'vacuum_multixact_freeze_table_age'
        ]

        # Query all PostgreSQL settings for autovacuum using last_over_time
        settings_query = f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        result = self.query_instant(settings_query)

        autovacuum_data = {}
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                setting_name = item['metric'].get('setting_name', 'unknown')

                # Filter for autovacuum and vacuum settings
                if setting_name in autovacuum_settings:
                    setting_value = item['metric'].get('setting_value', '')
                    category = item['metric'].get('category', 'Autovacuum')
                    unit = item['metric'].get('unit', '')
                    context = item['metric'].get('context', '')
                    vartype = item['metric'].get('vartype', '')

                    autovacuum_data[setting_name] = {
                        "setting": setting_value,
                        "unit": unit,
                        "category": category,
                        "context": context,
                        "vartype": vartype,
                        "pretty_value": self.format_setting_value(setting_name, setting_value, unit)
                    }

        return self.format_report_data("F001", autovacuum_data, node_name, postgres_version=self._get_postgres_version_info(cluster, node_name))

    def generate_f005_btree_bloat_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate F005 Autovacuum: Btree Index Bloat (Estimated) report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing btree index bloat information
        """
        logger.info("Generating F005 Autovacuum: Btree Index Bloat (Estimated) report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)

        # Get database sizes
        db_sizes_query = f'last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        db_sizes_result = self.query_instant(db_sizes_query)
        database_sizes = {}

        if db_sizes_result.get('status') == 'success' and db_sizes_result.get('data', {}).get('result'):
            for result in db_sizes_result['data']['result']:
                db_name = result['metric'].get('datname', 'unknown')
                size_bytes = float(result['value'][1])
                database_sizes[db_name] = size_bytes

        bloated_indexes_by_db = {}
        for db_name in databases:
            # Fetch last vacuum timestamp per table (from pg_stat_all_tables) so we can attach it to indexes.
            last_vacuum_query = (
                f'last_over_time(pgwatch_pg_stat_all_tables_last_vacuum'
                f'{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])'
            )
            last_vacuum_result = self.query_instant(last_vacuum_query)
            last_vacuum_by_table: Dict[str, float] = {}
            if last_vacuum_result.get('status') == 'success' and last_vacuum_result.get('data', {}).get('result'):
                for item in last_vacuum_result['data']['result']:
                    metric = item.get('metric', {})
                    schema_name = (
                        metric.get('schemaname')
                        or metric.get('tag_schemaname')
                        or 'unknown'
                    )
                    # pg_stat_all_tables uses relname, but be defensive in case of label differences.
                    relname = (
                        metric.get('relname')
                        or metric.get('tag_relname')
                        or metric.get('tblname')
                        or metric.get('tag_tblname')
                        or metric.get('table_name')
                        or 'unknown'
                    )
                    key = f"{schema_name}.{relname}"
                    value = float(item['value'][1]) if item.get('value') else 0
                    last_vacuum_by_table[key] = value

            # Fetch table sizes from pg_class as a fallback if pg_btree_bloat_table_size_mib is unavailable.
            table_sizes_query = (
                f'last_over_time(pgwatch_pg_class_relation_size_bytes'
                f'{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}", relkind="r"}}[3h])'
            )
            table_sizes_result = self.query_instant(table_sizes_query)
            table_size_by_table: Dict[str, float] = {}
            if table_sizes_result.get('status') == 'success' and table_sizes_result.get('data', {}).get('result'):
                for item in table_sizes_result['data']['result']:
                    metric = item.get('metric', {}) or {}
                    schema_name = (
                        metric.get('schemaname')
                        or metric.get('tag_schemaname')
                        or 'unknown'
                    )
                    relname = (
                        metric.get('relname')
                        or metric.get('tag_relname')
                        or metric.get('tblname')
                        or metric.get('tag_tblname')
                        or metric.get('table_name')
                        or 'unknown'
                    )
                    key = f"{schema_name}.{relname}"
                    value = float(item['value'][1]) if item.get('value') else 0
                    table_size_by_table[key] = value

            # Query btree bloat using multiple metrics for each database with last_over_time [1d]
            bloat_queries = {
                # Backward/forward compatible:
                # - Older pgwatch configs may expose bytes gauges (real_size, table_size)
                # - Newer configs expose MiB gauges (real_size_mib, table_size_mib)
                'real_size_mib': f'last_over_time(pgwatch_pg_btree_bloat_real_size_mib{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'real_size': f'last_over_time(pgwatch_pg_btree_bloat_real_size{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'table_size_mib': f'last_over_time(pgwatch_pg_btree_bloat_table_size_mib{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'table_size': f'last_over_time(pgwatch_pg_btree_bloat_table_size{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'extra_size': f'last_over_time(pgwatch_pg_btree_bloat_extra_size{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'extra_pct': f'last_over_time(pgwatch_pg_btree_bloat_extra_pct{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'fillfactor': f'last_over_time(pgwatch_pg_btree_bloat_fillfactor{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'bloat_size': f'last_over_time(pgwatch_pg_btree_bloat_bloat_size{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'bloat_pct': f'last_over_time(pgwatch_pg_btree_bloat_bloat_pct{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
            }

            bloated_indexes = {}

            for metric_type, query in bloat_queries.items():
                result = self.query_instant(query)
                if result.get('status') == 'success' and result.get('data', {}).get('result'):
                    for item in result['data']['result']:
                        metric = item.get('metric', {}) or {}
                        schema_name = (
                            metric.get('schemaname')
                            or metric.get('tag_schemaname')
                            or 'unknown'
                        )
                        table_name = (
                            metric.get('tblname')
                            or metric.get('tag_tblname')
                            or metric.get('relname')
                            or metric.get('tag_relname')
                            or metric.get('table_name')
                            or 'unknown'
                        )
                        index_name = (
                            metric.get('idxname')
                            or metric.get('tag_idxname')
                            or metric.get('index_name')
                            or 'unknown'
                        )

                        index_key = f"{schema_name}.{table_name}.{index_name}"

                        if index_key not in bloated_indexes:
                            bloated_indexes[index_key] = {
                                "schema_name": schema_name,
                                "table_name": table_name,
                                "index_name": index_name,
                                "real_size_mib": 0,
                                "table_size_mib": 0,
                                "real_size": 0,   # bytes (from bytes gauge or derived from MiB)
                                "table_size": 0,  # bytes (from bytes gauge or derived from MiB or fallback)
                                "extra_size": 0,
                                "extra_pct": 0,
                                "fillfactor": 0,
                                "bloat_size": 0,
                                "bloat_pct": 0,
                                "last_vacuum": 0,
                            }

                        value = float(item['value'][1]) if item.get('value') else 0
                        bloated_indexes[index_key][metric_type] = value
            
            # Skip databases with no bloat data
            if not bloated_indexes:
                continue

            # Convert to list and add pretty formatting
            bloated_indexes_list = []
            total_bloat_size = 0

            for index_data in bloated_indexes.values():
                key = f"{index_data.get('schema_name', 'unknown')}.{index_data.get('table_name', 'unknown')}"
                last_vacuum_epoch = float(last_vacuum_by_table.get(key, 0) or 0)
                index_data['last_vacuum_epoch'] = last_vacuum_epoch
                index_data['last_vacuum'] = self.format_epoch_timestamp(last_vacuum_epoch)
                # Sizes are bytes in the report output.
                # Prefer bytes gauges if present, otherwise convert from MiB.
                real_size = float(index_data.get('real_size', 0) or 0)
                if real_size <= 0:
                    real_size_mib = float(index_data.get('real_size_mib', 0) or 0)
                    real_size = real_size_mib * 1024 * 1024 if real_size_mib > 0 else 0
                index_data['real_size'] = int(real_size)
                index_data.pop('real_size_mib', None)

                table_size = float(index_data.get('table_size', 0) or 0)
                if table_size <= 0:
                    table_size_mib = float(index_data.get('table_size_mib', 0) or 0)
                    table_size = table_size_mib * 1024 * 1024 if table_size_mib > 0 else 0
                if table_size <= 0:
                    table_size = float(table_size_by_table.get(key, 0) or 0)
                index_data['table_size'] = int(table_size)
                index_data.pop('table_size_mib', None)

                index_data['real_size_pretty'] = self.format_bytes(index_data['real_size'])
                index_data['table_size_pretty'] = self.format_bytes(index_data['table_size'])
                index_data['extra_size_pretty'] = self.format_bytes(index_data['extra_size'])
                index_data['bloat_size_pretty'] = self.format_bytes(index_data['bloat_size'])

                bloated_indexes_list.append(index_data)
                total_bloat_size += index_data['bloat_size']

            # Sort by bloat percentage descending
            bloated_indexes_list.sort(key=lambda x: x['bloat_pct'], reverse=True)

            db_size_bytes = database_sizes.get(db_name, 0)
            bloated_indexes_by_db[db_name] = {
                "bloated_indexes": bloated_indexes_list,
                "total_count": len(bloated_indexes_list),
                "total_bloat_size_bytes": total_bloat_size,
                "total_bloat_size_pretty": self.format_bytes(total_bloat_size),
                "database_size_bytes": db_size_bytes,
                "database_size_pretty": self.format_bytes(db_size_bytes)
            }

        return self.format_report_data(
            "F005",
            bloated_indexes_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_g001_memory_settings_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[
        str, Any]:
        """
        Generate G001 Memory-related Settings report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing memory-related settings information
        """
        logger.info("Generating G001 Memory-related Settings report...")

        # Define memory-related settings
        memory_settings = [
            'shared_buffers',
            'work_mem',
            'maintenance_work_mem',
            'effective_cache_size',
            'autovacuum_work_mem',
            'max_wal_size',
            'min_wal_size',
            'wal_buffers',
            'checkpoint_completion_target',
            'max_connections',
            'max_prepared_transactions',
            'max_locks_per_transaction',
            'max_pred_locks_per_transaction',
            'max_pred_locks_per_relation',
            'max_pred_locks_per_page',
            'logical_decoding_work_mem',
            'hash_mem_multiplier',
            'temp_buffers',
            'shared_preload_libraries',
            'dynamic_shared_memory_type',
            'huge_pages',
            'max_files_per_process',
            'max_stack_depth'
        ]

        # Query all PostgreSQL settings for memory-related settings using last_over_time
        settings_query = f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        result = self.query_instant(settings_query)

        memory_data = {}
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                setting_name = item['metric'].get('setting_name', '')
                
                # Skip if no setting name
                if not setting_name:
                    continue

                # Filter for memory-related settings
                if setting_name in memory_settings:
                    setting_value = item['metric'].get('setting_value', '')
                    category = item['metric'].get('category', 'Memory')
                    unit = item['metric'].get('unit', '')
                    context = item['metric'].get('context', '')
                    vartype = item['metric'].get('vartype', '')

                    memory_data[setting_name] = {
                        "setting": setting_value,
                        "unit": unit,
                        "category": category,
                        "context": context,
                        "vartype": vartype,
                        "pretty_value": self.format_setting_value(setting_name, setting_value, unit)
                    }
        else:
            logger.warning(f"G001 - No settings data returned for cluster={cluster}, node_name={node_name}")

        # Calculate some memory usage estimates and recommendations
        memory_analysis = self._analyze_memory_settings(memory_data)

        return self.format_report_data(
            "G001",
            {
                "settings": memory_data,
                "analysis": memory_analysis,
            },
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def _analyze_memory_settings(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze memory settings and provide estimates and recommendations.
        
        Args:
            memory_data: Dictionary of memory settings
            
        Returns:
            Dictionary containing memory analysis
        """
        analysis = {
            "estimated_total_memory_usage": {}
        }

        try:
            # Extract key memory values for analysis
            shared_buffers = self._parse_memory_value(memory_data.get('shared_buffers', {}).get('setting', '128MB'))
            work_mem = self._parse_memory_value(memory_data.get('work_mem', {}).get('setting', '4MB'))
            maintenance_work_mem = self._parse_memory_value(
                memory_data.get('maintenance_work_mem', {}).get('setting', '64MB'))
            effective_cache_size = self._parse_memory_value(
                memory_data.get('effective_cache_size', {}).get('setting', '4GB'))
            max_connections = int(memory_data.get('max_connections', {}).get('setting', '100'))
            wal_buffers = self._parse_memory_value(memory_data.get('wal_buffers', {}).get('setting', '16MB'))

            # Calculate estimated memory usage
            shared_memory = shared_buffers + wal_buffers
            potential_work_mem_usage = work_mem * max_connections  # Worst case scenario

            analysis["estimated_total_memory_usage"] = {
                "shared_buffers_bytes": shared_buffers,
                "shared_buffers_pretty": self.format_bytes(shared_buffers),
                "wal_buffers_bytes": wal_buffers,
                "wal_buffers_pretty": self.format_bytes(wal_buffers),
                "shared_memory_total_bytes": shared_memory,
                "shared_memory_total_pretty": self.format_bytes(shared_memory),
                "work_mem_per_connection_bytes": work_mem,
                "work_mem_per_connection_pretty": self.format_bytes(work_mem),
                "max_work_mem_usage_bytes": potential_work_mem_usage,
                "max_work_mem_usage_pretty": self.format_bytes(potential_work_mem_usage),
                "maintenance_work_mem_bytes": maintenance_work_mem,
                "maintenance_work_mem_pretty": self.format_bytes(maintenance_work_mem),
                "effective_cache_size_bytes": effective_cache_size,
                "effective_cache_size_pretty": self.format_bytes(effective_cache_size)
            }

            # Generate recommendations                            
        except (ValueError, TypeError):
            # If parsing fails, return empty analysis
            analysis["estimated_total_memory_usage"] = {}

        return analysis

    def _parse_memory_value(self, value: str) -> int:
        """
        Parse a PostgreSQL memory value string to bytes.
        
        Args:
            value: Memory value string (e.g., "128MB", "4GB", "8192")
            
        Returns:
            Memory value in bytes
        """
        if not value or value == '-1':
            return 0

        value = str(value).strip().upper()

        # Handle unit suffixes
        if value.endswith('TB'):
            return int(float(value[:-2]) * 1024 * 1024 * 1024 * 1024)
        elif value.endswith('GB'):
            return int(float(value[:-2]) * 1024 * 1024 * 1024)
        elif value.endswith('MB'):
            return int(float(value[:-2]) * 1024 * 1024)
        elif value.endswith('KB'):
            return int(float(value[:-2]) * 1024)
        elif value.endswith('B'):
            return int(float(value[:-1]))
        else:
            # Assume it's in the PostgreSQL default unit (typically 8KB blocks for some settings)
            try:
                numeric_value = int(value)
                # For most memory settings, bare numbers are in KB or 8KB blocks
                # This is a simplified assumption - in reality it depends on the specific setting
                return numeric_value * 1024  # Assume KB if no unit specified
            except ValueError:
                return 0

    def generate_f004_heap_bloat_report(self, cluster: str = "local", node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate F004 Autovacuum: Heap Bloat (Estimated) report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            Dictionary containing heap bloat information
        """
        logger.info("Generating F004 Autovacuum: Heap Bloat (Estimated) report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("F004 - No databases found")

        # Get database sizes
        db_sizes_query = f'last_over_time(pgwatch_db_size_size_b{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        db_sizes_result = self.query_instant(db_sizes_query)
        database_sizes = {}

        if db_sizes_result.get('status') == 'success' and db_sizes_result.get('data', {}).get('result'):
            for result in db_sizes_result['data']['result']:
                db_name = result['metric'].get('datname', 'unknown')
                size_bytes = float(result['value'][1])
                database_sizes[db_name] = size_bytes

        bloated_tables_by_db = {}
        for db_name in databases:
            # Fetch last vacuum timestamp per table (from pg_stat_all_tables).
            # Note: prefer `relname`, but be defensive since other parts of the codebase / configs
            # sometimes use `tblname`.
            last_vacuum_query = (
                f'last_over_time(pgwatch_pg_stat_all_tables_last_vacuum'
                f'{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])'
            )
            last_vacuum_result = self.query_instant(last_vacuum_query)
            last_vacuum_by_table: Dict[str, float] = {}
            if last_vacuum_result.get('status') == 'success' and last_vacuum_result.get('data', {}).get('result'):
                for item in last_vacuum_result['data']['result']:
                    metric = item.get('metric', {})
                    schema_name = (
                        metric.get('schemaname')
                        or metric.get('tag_schemaname')
                        or 'unknown'
                    )
                    relname = (
                        metric.get('relname')
                        or metric.get('tag_relname')
                        or metric.get('tblname')
                        or metric.get('tag_tblname')
                        or metric.get('table_name')
                        or 'unknown'
                    )
                    key = f"{schema_name}.{relname}"
                    value = float(item['value'][1]) if item.get('value') else 0
                    last_vacuum_by_table[key] = value

            # Query table bloat using multiple metrics for each database
            # Try with 10h window first, then fall back to instant query
            bloat_queries = {
                # pgwatch publishes "real size" in MiB (real_size_mib). We keep 'real_size' in the
                # output as a backwards-compatible alias but it is based on MiB.
                'real_size_mib': f'last_over_time(pgwatch_pg_table_bloat_real_size_mib{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'extra_size': f'last_over_time(pgwatch_pg_table_bloat_extra_size{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'extra_pct': f'last_over_time(pgwatch_pg_table_bloat_extra_pct{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'fillfactor': f'last_over_time(pgwatch_pg_table_bloat_fillfactor{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'bloat_size': f'last_over_time(pgwatch_pg_table_bloat_bloat_size{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
                'bloat_pct': f'last_over_time(pgwatch_pg_table_bloat_bloat_pct{{cluster="{cluster}", node_name="{node_name}", datname="{db_name}"}}[3h])',
            }

            bloated_tables = {}
            for metric_type, query in bloat_queries.items():
                result = self.query_instant(query)
                if result.get('status') == 'success' and result.get('data', {}).get('result'):
                    for item in result['data']['result']:
                        schema_name = item['metric'].get('schemaname', 'unknown')
                        table_name = item['metric'].get('tblname', 'unknown')

                        table_key = f"{schema_name}.{table_name}"

                        if table_key not in bloated_tables:
                            bloated_tables[table_key] = {
                                "schema_name": schema_name,
                                "table_name": table_name,
                                # Stored temporarily as MiB because pgwatch publishes real_size_mib.
                                # We'll convert it to bytes for report output.
                                "real_size": 0,
                                "extra_size": 0,
                                "extra_pct": 0,
                                "fillfactor": 0,
                                "bloat_size": 0,
                                "bloat_pct": 0,
                                "last_vacuum": 0,
                            }

                        value = float(item['value'][1]) if item.get('value') else 0
                        bloated_tables[table_key][metric_type] = value
                else:
                    if metric_type == 'real_size_mib':  # Only log once per database
                        logger.warning(f"F004 - No bloat data for database {db_name}, metric {metric_type}")
            
            # Skip databases with no bloat data
            if not bloated_tables:
                continue

            # Convert to list and add pretty formatting
            bloated_tables_list = []
            total_bloat_size = 0

            for table_data in bloated_tables.values():
                # Normalize real size: Prometheus provides it in MiB (real_size_mib), but the report
                # should expose real_size in bytes.
                real_size_mib = float(table_data.get('real_size_mib', 0) or 0)
                table_data['real_size'] = int(real_size_mib * 1024 * 1024)
                # Remove intermediate field so it's not part of the report payload.
                table_data.pop('real_size_mib', None)
                # Attach last vacuum timestamp (epoch seconds) from pg_stat_all_tables.
                key = f"{table_data.get('schema_name', 'unknown')}.{table_data.get('table_name', 'unknown')}"
                last_vacuum_epoch = float(last_vacuum_by_table.get(key, 0) or 0)
                table_data['last_vacuum_epoch'] = last_vacuum_epoch
                table_data['last_vacuum'] = self.format_epoch_timestamp(last_vacuum_epoch)
                table_data['real_size_pretty'] = self.format_bytes(table_data['real_size'])
                table_data['extra_size_pretty'] = self.format_bytes(table_data['extra_size'])
                table_data['bloat_size_pretty'] = self.format_bytes(table_data['bloat_size'])

                bloated_tables_list.append(table_data)
                total_bloat_size += table_data['bloat_size']

            # Sort by bloat percentage descending
            bloated_tables_list.sort(key=lambda x: x['bloat_pct'], reverse=True)

            db_size_bytes = database_sizes.get(db_name, 0)
            bloated_tables_by_db[db_name] = {
                "bloated_tables": bloated_tables_list,
                "total_count": len(bloated_tables_list),
                "total_bloat_size_bytes": total_bloat_size,
                "total_bloat_size_pretty": self.format_bytes(total_bloat_size),
                "database_size_bytes": db_size_bytes,
                "database_size_pretty": self.format_bytes(db_size_bytes)
            }

        return self.format_report_data(
            "F004",
            bloated_tables_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k001_query_calls_report(self, cluster: str = "local", node_name: str = "node-01",
                                         time_range_minutes: int = 60, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate K001 Globally Aggregated Query Metrics report (sorted by calls).
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing query metrics sorted by calls
        """
        logger.info("Generating K001 Globally Aggregated Query Metrics report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("K001 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            metric_name = "pgwatch_pg_stat_statements_calls"
            
            for db_name in databases:
                logger.info(f"K001: Processing database {db_name} (hourly mode)...")
                
                per_query, other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, metric_name, hours=hours
                )
                
                if not per_query and sum(other) == 0:
                    logger.warning(f"K001 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate total calls per query across all hours
                query_totals = []
                for queryid, hourly_values in per_query.items():
                    total_calls = sum(hourly_values)
                    query_totals.append({
                        "queryid": queryid,
                        "total_calls": total_calls,
                        "hourly_calls": hourly_values
                    })
                
                # Sort by total calls (descending)
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_calls', 0), reverse=True)
                
                # Calculate totals
                total_calls = sum(q.get('total_calls', 0) for q in sorted_metrics) + sum(other)
                
                queries_by_db[db_name] = {
                    "query_metrics": sorted_metrics,
                    "other_calls_hourly": other,
                    "summary": {
                        "total_queries_tracked": len(sorted_metrics),
                        "total_calls": total_calls,
                        "total_calls_tracked_queries": sum(q.get('total_calls', 0) for q in sorted_metrics),
                        "total_calls_other": sum(other),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline
                    }
                }
        else:
            # Fallback to original logic for sub-hourly or when explicitly disabled
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            for db_name in databases:
                logger.info(f"K001: Processing database {db_name}...")
                # Get pg_stat_statements metrics for this database
                query_metrics = self._get_pgss_metrics_data_by_db(cluster, node_name, db_name, start_time, end_time)

                if not query_metrics:
                    logger.warning(f"K001 - No query metrics returned for database {db_name}")

                # Sort by calls (descending)
                sorted_metrics = sorted(query_metrics, key=lambda x: x.get('calls', 0), reverse=True)

                # Calculate totals for this database
                total_calls = sum(q.get('calls', 0) for q in sorted_metrics)
                total_time = sum(q.get('total_time', 0) for q in sorted_metrics)
                total_rows = sum(q.get('rows', 0) for q in sorted_metrics)

                queries_by_db[db_name] = {
                    "query_metrics": sorted_metrics,
                    "summary": {
                        "total_queries": len(sorted_metrics),
                        "total_calls": total_calls,
                        "total_time_ms": total_time,
                        "total_rows": total_rows,
                        "time_range_minutes": time_range_minutes,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat()
                    }
                }

        return self.format_report_data(
            "K001",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k003_top_queries_report(self, cluster: str = "local", node_name: str = "node-01",
                                         time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate K003 Top Queries by total_time (exec + plan) report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by total execution time (exec + plan)
        """
        logger.info("Generating K003 Top Queries by total_time (exec + plan) report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("K003 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            
            for db_name in databases:
                logger.info(f"K003: Processing database {db_name} (hourly mode)...")
                
                # Get exec time
                exec_per_query, exec_other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, "pgwatch_pg_stat_statements_exec_time_total", hours=hours
                )
                
                # Get plan time (might not be available in older PG versions)
                plan_per_query, plan_other, _ = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, "pgwatch_pg_stat_statements_plan_time_total", hours=hours
                )
                
                # Check if plan time is actually non-zero (not just if data exists)
                total_plan_time = sum(sum(values) for values in plan_per_query.values()) + sum(plan_other)
                plan_time_available = total_plan_time > 0
                
                if not exec_per_query and sum(exec_other) == 0:
                    logger.warning(f"K003 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Combine exec and plan time per query across all hours
                all_queryids = set(exec_per_query.keys()) | set(plan_per_query.keys())
                query_totals = []
                
                for queryid in all_queryids:
                    exec_values = exec_per_query.get(queryid, [0] * hours)
                    plan_values = plan_per_query.get(queryid, [0] * hours)
                    
                    # Combine exec + plan for each hour
                    hourly_total_time = [e + p for e, p in zip(exec_values, plan_values)]
                    total_time = sum(hourly_total_time)
                    total_exec_time = sum(exec_values)
                    total_plan_time = sum(plan_values)
                    
                    query_totals.append({
                        "queryid": queryid,
                        "total_time_ms": total_time,
                        "total_exec_time_ms": total_exec_time,
                        "total_plan_time_ms": total_plan_time,
                        "hourly_time_ms": hourly_total_time,
                        "hourly_exec_time_ms": exec_values,
                        "hourly_plan_time_ms": plan_values if plan_time_available else None
                    })
                
                # Sort by total_time (descending) and limit to top N
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_time_ms', 0), reverse=True)[:limit]
                
                # Calculate other time (exec + plan)
                other_time_hourly = [e + p for e, p in zip(exec_other, plan_other)]
                
                # Calculate totals
                total_time = sum(q.get('total_time_ms', 0) for q in sorted_metrics) + sum(other_time_hourly)
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_time_hourly": other_time_hourly,
                    "other_exec_time_hourly": exec_other,
                    "other_plan_time_hourly": plan_other if plan_time_available else None,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_time_ms": total_time,
                        "total_time_tracked_queries_ms": sum(q.get('total_time_ms', 0) for q in sorted_metrics),
                        "total_time_other_ms": sum(other_time_hourly),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit,
                        "plan_time_available": plan_time_available,
                        "note": "Includes both exec and plan time" if plan_time_available else "Plan time unavailable, showing exec time only"
                    }
                }
        else:
            # Fallback to original logic for sub-hourly or when explicitly disabled
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            for db_name in databases:
                logger.info(f"K003: Processing database {db_name}...")
                # Get pg_stat_statements metrics for this database
                query_metrics = self._get_pgss_metrics_data_by_db(cluster, node_name, db_name, start_time, end_time)

                if not query_metrics:
                    logger.warning(f"K003 - No query metrics returned for database {db_name}")

                # Sort by total_time (descending) and limit to top N per database
                sorted_metrics = sorted(query_metrics, key=lambda x: x.get('total_time', 0), reverse=True)[:limit]

                # Calculate totals for the top queries in this database
                total_calls = sum(q.get('calls', 0) for q in sorted_metrics)
                total_time = sum(q.get('total_time', 0) for q in sorted_metrics)
                total_rows = sum(q.get('rows', 0) for q in sorted_metrics)

                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_calls": total_calls,
                        "total_time_ms": total_time,
                        "total_rows": total_rows,
                        "time_range_minutes": time_range_minutes,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "limit": limit
                    }
                }

        return self.format_report_data(
            "K003",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_m001_mean_time_report(self, cluster: str = "local", node_name: str = "node-01",
                                       time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate M001 Top Queries by mean execution time report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by mean execution time
        """
        logger.info("Generating M001 Top Queries by mean execution time report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("M001 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            
            for db_name in databases:
                logger.info(f"M001: Processing database {db_name} (hourly mode)...")
                
                # Get both time and calls metrics
                time_per_query, time_other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, "pgwatch_pg_stat_statements_exec_time_total", hours=hours
                )
                calls_per_query, calls_other, _ = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, "pgwatch_pg_stat_statements_calls", hours=hours
                )
                
                if not time_per_query and sum(time_other) == 0:
                    logger.warning(f"M001 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate mean time per query across all hours
                query_means = []
                for queryid in time_per_query.keys():
                    total_time = sum(time_per_query[queryid])
                    total_calls = sum(calls_per_query.get(queryid, [0] * hours))
                    
                    if total_calls > 0:
                        mean_time = total_time / total_calls
                        query_means.append({
                            "queryid": queryid,
                            "mean_time_ms": mean_time,
                            "total_time_ms": total_time,
                            "total_calls": total_calls,
                            "hourly_time_ms": time_per_query[queryid],
                            "hourly_calls": calls_per_query.get(queryid, [0] * hours)
                        })
                
                # Sort by mean_time (descending) and limit to top N
                sorted_metrics = sorted(query_means, key=lambda x: x.get('mean_time_ms', 0), reverse=True)[:limit]
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_time_hourly": time_other,
                    "other_calls_hourly": calls_other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_time_tracked_queries_ms": sum(q.get('total_time_ms', 0) for q in sorted_metrics),
                        "total_calls_tracked_queries": sum(q.get('total_calls', 0) for q in sorted_metrics),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback to original logic for sub-hourly or when explicitly disabled
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            for db_name in databases:
                logger.info(f"M001: Processing database {db_name}...")
                # Get pg_stat_statements metrics for this database
                query_metrics = self._get_pgss_metrics_data_by_db(cluster, node_name, db_name, start_time, end_time)

                if not query_metrics:
                    logger.warning(f"M001 - No query metrics returned for database {db_name}")

                # Calculate mean execution time for each query
                queries_with_mean = []
                for q in query_metrics:
                    calls = q.get('calls', 0)
                    total_time = q.get('total_time', 0)
                    if calls > 0:
                        mean_time = total_time / calls
                        q['mean_time'] = mean_time
                        queries_with_mean.append(q)

                # Sort by mean_time (descending) and limit to top N per database
                sorted_metrics = sorted(queries_with_mean, key=lambda x: x.get('mean_time', 0), reverse=True)[:limit]

                # Calculate totals for the top queries in this database
                total_calls = sum(q.get('calls', 0) for q in sorted_metrics)
                total_time = sum(q.get('total_time', 0) for q in sorted_metrics)
                total_rows = sum(q.get('rows', 0) for q in sorted_metrics)

                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_calls": total_calls,
                        "total_time_ms": total_time,
                        "total_rows": total_rows,
                        "time_range_minutes": time_range_minutes,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "limit": limit
                    }
                }

        return self.format_report_data(
            "M001",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_m002_rows_report(self, cluster: str = "local", node_name: str = "node-01",
                                  time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate M002 Top Queries by rows (I/O intensity) report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by rows processed
        """
        logger.info("Generating M002 Top Queries by rows report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("M002 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            metric_name = "pgwatch_pg_stat_statements_rows"
            
            for db_name in databases:
                logger.info(f"M002: Processing database {db_name} (hourly mode)...")
                
                per_query, other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, metric_name, hours=hours
                )
                
                if not per_query and sum(other) == 0:
                    logger.warning(f"M002 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate total rows per query across all hours
                query_totals = []
                for queryid, hourly_values in per_query.items():
                    total_rows = sum(hourly_values)
                    query_totals.append({
                        "queryid": queryid,
                        "total_rows": total_rows,
                        "hourly_rows": hourly_values
                    })
                
                # Sort by total_rows (descending) and limit to top N
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_rows', 0), reverse=True)[:limit]
                
                # Calculate totals
                total_rows = sum(q.get('total_rows', 0) for q in sorted_metrics) + sum(other)
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_rows_hourly": other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_rows": total_rows,
                        "total_rows_tracked_queries": sum(q.get('total_rows', 0) for q in sorted_metrics),
                        "total_rows_other": sum(other),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback to original logic for sub-hourly or when explicitly disabled
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            for db_name in databases:
                logger.info(f"M002: Processing database {db_name}...")
                # Get pg_stat_statements metrics for this database
                query_metrics = self._get_pgss_metrics_data_by_db(cluster, node_name, db_name, start_time, end_time)

                if not query_metrics:
                    logger.warning(f"M002 - No query metrics returned for database {db_name}")

                # Sort by rows (descending) and limit to top N per database
                sorted_metrics = sorted(query_metrics, key=lambda x: x.get('rows', 0), reverse=True)[:limit]

                # Calculate totals for the top queries in this database
                total_calls = sum(q.get('calls', 0) for q in sorted_metrics)
                total_time = sum(q.get('total_time', 0) for q in sorted_metrics)
                total_rows = sum(q.get('rows', 0) for q in sorted_metrics)

                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_calls": total_calls,
                        "total_time_ms": total_time,
                        "total_rows": total_rows,
                        "time_range_minutes": time_range_minutes,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "limit": limit
                    }
                }

        return self.format_report_data(
            "M002",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_m003_io_time_report(self, cluster: str = "local", node_name: str = "node-01",
                                     time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate M003 Top Queries by I/O time report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by total I/O time
        """
        logger.info("Generating M003 Top Queries by I/O time report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("M003 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            
            for db_name in databases:
                logger.info(f"M003: Processing database {db_name} (hourly mode)...")
                
                # Get both read and write I/O time metrics
                read_per_query, read_other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, "pgwatch_pg_stat_statements_block_read_total", hours=hours
                )
                write_per_query, write_other, _ = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, "pgwatch_pg_stat_statements_block_write_total", hours=hours
                )
                
                if not read_per_query and not write_per_query and sum(read_other) == 0 and sum(write_other) == 0:
                    logger.warning(f"M003 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Combine read and write times, calculate total I/O time per query
                all_queryids = set(read_per_query.keys()) | set(write_per_query.keys())
                query_io_totals = []
                
                for queryid in all_queryids:
                    read_values = read_per_query.get(queryid, [0] * hours)
                    write_values = write_per_query.get(queryid, [0] * hours)
                    
                    # Combine read and write for each hour
                    hourly_io_time = [r + w for r, w in zip(read_values, write_values)]
                    total_io_time = sum(hourly_io_time)
                    
                    query_io_totals.append({
                        "queryid": queryid,
                        "total_io_time_ms": total_io_time,
                        "total_read_time_ms": sum(read_values),
                        "total_write_time_ms": sum(write_values),
                        "hourly_io_time_ms": hourly_io_time,
                        "hourly_read_time_ms": read_values,
                        "hourly_write_time_ms": write_values
                    })
                
                # Sort by total_io_time (descending) and limit to top N
                sorted_metrics = sorted(query_io_totals, key=lambda x: x.get('total_io_time_ms', 0), reverse=True)[:limit]
                
                # Calculate other I/O time
                other_io_time_hourly = [r + w for r, w in zip(read_other, write_other)]
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_io_time_hourly": other_io_time_hourly,
                    "other_read_time_hourly": read_other,
                    "other_write_time_hourly": write_other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_io_time_tracked_queries_ms": sum(q.get('total_io_time_ms', 0) for q in sorted_metrics),
                        "total_io_time_other_ms": sum(other_io_time_hourly),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback to original logic for sub-hourly or when explicitly disabled
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            for db_name in databases:
                logger.info(f"M003: Processing database {db_name}...")
                # Get pg_stat_statements metrics for this database
                query_metrics = self._get_pgss_metrics_data_by_db(cluster, node_name, db_name, start_time, end_time)

                if not query_metrics:
                    logger.warning(f"M003 - No query metrics returned for database {db_name}")

                # Calculate total I/O time for each query
                queries_with_io_time = []
                for q in query_metrics:
                    blk_read_time = q.get('blk_read_time', 0)
                    blk_write_time = q.get('blk_write_time', 0)
                    total_io_time = blk_read_time + blk_write_time
                    q['total_io_time'] = total_io_time
                    queries_with_io_time.append(q)

                # Sort by total_io_time (descending) and limit to top N per database
                sorted_metrics = sorted(queries_with_io_time, key=lambda x: x.get('total_io_time', 0), reverse=True)[:limit]

                # Calculate totals for the top queries in this database
                total_calls = sum(q.get('calls', 0) for q in sorted_metrics)
                total_time = sum(q.get('total_time', 0) for q in sorted_metrics)
                total_rows = sum(q.get('rows', 0) for q in sorted_metrics)
                total_io_time = sum(q.get('total_io_time', 0) for q in sorted_metrics)

                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_calls": total_calls,
                        "total_time_ms": total_time,
                        "total_rows": total_rows,
                        "total_io_time_ms": total_io_time,
                        "time_range_minutes": time_range_minutes,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "limit": limit
                    }
                }

        return self.format_report_data(
            "M003",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k004_temp_bytes_report(self, cluster: str = "local", node_name: str = "node-01",
                                        time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate K004 Top Queries by temp bytes written report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by temp bytes written
        """
        logger.info("Generating K004 Top Queries by temp bytes written report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("K004 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            metric_name = "pgwatch_pg_stat_statements_temp_bytes_written"
            
            for db_name in databases:
                logger.info(f"K004: Processing database {db_name} (hourly mode)...")
                
                per_query, other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, metric_name, hours=hours
                )
                
                if not per_query and sum(other) == 0:
                    logger.warning(f"K004 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate total temp bytes per query across all hours
                query_totals = []
                for queryid, hourly_values in per_query.items():
                    total_bytes = sum(hourly_values)
                    query_totals.append({
                        "queryid": queryid,
                        "total_temp_bytes": total_bytes,
                        "hourly_temp_bytes": hourly_values
                    })
                
                # Sort by total_temp_bytes (descending) and limit to top N
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_temp_bytes', 0), reverse=True)[:limit]
                
                # Calculate totals
                total_bytes = sum(q.get('total_temp_bytes', 0) for q in sorted_metrics) + sum(other)
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_temp_bytes_hourly": other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_temp_bytes": total_bytes,
                        "total_temp_bytes_tracked_queries": sum(q.get('total_temp_bytes', 0) for q in sorted_metrics),
                        "total_temp_bytes_other": sum(other),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback for sub-hourly (not typically needed)
            pass

        return self.format_report_data(
            "K004",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k005_wal_bytes_report(self, cluster: str = "local", node_name: str = "node-01",
                                       time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate K005 Top Queries by WAL generation report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by WAL bytes generated
        """
        logger.info("Generating K005 Top Queries by WAL generation report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("K005 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            metric_name = "pgwatch_pg_stat_statements_wal_bytes"
            
            for db_name in databases:
                logger.info(f"K005: Processing database {db_name} (hourly mode)...")
                
                per_query, other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, metric_name, hours=hours
                )
                
                if not per_query and sum(other) == 0:
                    logger.warning(f"K005 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate total WAL bytes per query across all hours
                query_totals = []
                for queryid, hourly_values in per_query.items():
                    total_bytes = sum(hourly_values)
                    query_totals.append({
                        "queryid": queryid,
                        "total_wal_bytes": total_bytes,
                        "hourly_wal_bytes": hourly_values
                    })
                
                # Sort by total_wal_bytes (descending) and limit to top N
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_wal_bytes', 0), reverse=True)[:limit]
                
                # Calculate totals
                total_bytes = sum(q.get('total_wal_bytes', 0) for q in sorted_metrics) + sum(other)
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_wal_bytes_hourly": other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_wal_bytes": total_bytes,
                        "total_wal_bytes_tracked_queries": sum(q.get('total_wal_bytes', 0) for q in sorted_metrics),
                        "total_wal_bytes_other": sum(other),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback for sub-hourly (not typically needed)
            pass

        return self.format_report_data(
            "K005",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k006_shared_read_report(self, cluster: str = "local", node_name: str = "node-01",
                                         time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate K006 Top Queries by shared blocks read report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by shared blocks read
        """
        logger.info("Generating K006 Top Queries by shared blocks read report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("K006 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            metric_name = "pgwatch_pg_stat_statements_shared_bytes_read_total"
            
            for db_name in databases:
                logger.info(f"K006: Processing database {db_name} (hourly mode)...")
                
                per_query, other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, metric_name, hours=hours
                )
                
                if not per_query and sum(other) == 0:
                    logger.warning(f"K006 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate total shared read bytes per query across all hours
                query_totals = []
                for queryid, hourly_values in per_query.items():
                    total_bytes = sum(hourly_values)
                    query_totals.append({
                        "queryid": queryid,
                        "total_shared_read_bytes": total_bytes,
                        "hourly_shared_read_bytes": hourly_values
                    })
                
                # Sort by total_shared_read_bytes (descending) and limit to top N
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_shared_read_bytes', 0), reverse=True)[:limit]
                
                # Calculate totals
                total_bytes = sum(q.get('total_shared_read_bytes', 0) for q in sorted_metrics) + sum(other)
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_shared_read_bytes_hourly": other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_shared_read_bytes": total_bytes,
                        "total_shared_read_bytes_tracked_queries": sum(q.get('total_shared_read_bytes', 0) for q in sorted_metrics),
                        "total_shared_read_bytes_other": sum(other),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback for sub-hourly (not typically needed)
            pass

        return self.format_report_data(
            "K006",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k007_shared_hit_report(self, cluster: str = "local", node_name: str = "node-01",
                                        time_range_minutes: int = 60, limit: int = 100, use_hourly: bool = True) -> Dict[str, Any]:
        """
        Generate K007 Top Queries by shared blocks hit report.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)
            
        Returns:
            Dictionary containing top queries sorted by shared blocks hit
        """
        logger.info("Generating K007 Top Queries by shared blocks hit report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("K007 - No databases found")

        queries_by_db = {}
        
        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60
            metric_name = "pgwatch_pg_stat_statements_shared_bytes_hit_total"
            
            for db_name in databases:
                logger.info(f"K007: Processing database {db_name} (hourly mode)...")
                
                per_query, other, timeline = self._get_hourly_topk_pgss_data(
                    cluster, node_name, db_name, metric_name, hours=hours
                )
                
                if not per_query and sum(other) == 0:
                    logger.warning(f"K007 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data
                
                # Calculate total shared hit bytes per query across all hours
                query_totals = []
                for queryid, hourly_values in per_query.items():
                    total_bytes = sum(hourly_values)
                    query_totals.append({
                        "queryid": queryid,
                        "total_shared_hit_bytes": total_bytes,
                        "hourly_shared_hit_bytes": hourly_values
                    })
                
                # Sort by total_shared_hit_bytes (descending) and limit to top N
                sorted_metrics = sorted(query_totals, key=lambda x: x.get('total_shared_hit_bytes', 0), reverse=True)[:limit]
                
                # Calculate totals
                total_bytes = sum(q.get('total_shared_hit_bytes', 0) for q in sorted_metrics) + sum(other)
                
                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_shared_hit_bytes_hourly": other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_shared_hit_bytes": total_bytes,
                        "total_shared_hit_bytes_tracked_queries": sum(q.get('total_shared_hit_bytes', 0) for q in sorted_metrics),
                        "total_shared_hit_bytes_other": sum(other),
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit
                    }
                }
        else:
            # Fallback for sub-hourly (not typically needed)
            pass

        return self.format_report_data(
            "K007",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_k008_shared_hit_read_report(
        self,
        cluster: str = "local",
        node_name: str = "node-01",
        time_range_minutes: int = 60,
        limit: int = 100,
        use_hourly: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate K008 Top Queries by shared blocks (hit + read) report.

        Notes:
        - Our hourly topk utility (`_get_hourly_topk_pgss_data`) can only rank by a single metric.
          Here we fetch hit and read separately and then combine them in Python (similar to K003).

        Args:
            cluster: Cluster name
            node_name: Node name
            time_range_minutes: Time range in minutes for metrics collection (used when use_hourly=False)
            limit: Number of top queries to return (default: 100)
            use_hourly: Use hourly topk aggregation logic (default: True)

        Returns:
            Dictionary containing top queries sorted by (shared hit + shared read) bytes
        """
        logger.info("Generating K008 Top Queries by shared blocks (hit + read) report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)

        if not databases:
            logger.warning("K008 - No databases found")

        queries_by_db: Dict[str, Any] = {}

        if use_hourly and time_range_minutes >= 60:
            # Use hourly topk aggregation
            hours = time_range_minutes // 60

            hit_metric = "pgwatch_pg_stat_statements_shared_bytes_hit_total"
            read_metric = "pgwatch_pg_stat_statements_shared_bytes_read_total"

            for db_name in databases:
                logger.info(f"K008: Processing database {db_name} (hourly mode)...")

                per_query, other, timeline = self._get_hourly_topk_pgss_data_sum2(
                    cluster,
                    node_name,
                    db_name,
                    hit_metric,
                    read_metric,
                    hours=hours,
                )

                if not per_query and sum(other) == 0:
                    logger.warning(f"K008 - No query metrics returned for database {db_name}")
                    continue  # Skip databases with no data

                query_totals = []
                for queryid, hourly_total_bytes in per_query.items():
                    total_bytes = sum(hourly_total_bytes)
                    query_totals.append(
                        {
                            "queryid": queryid,
                            "total_shared_hit_read_bytes": total_bytes,
                            "hourly_shared_hit_read_bytes": hourly_total_bytes,
                        }
                    )

                # Sort by total_shared_hit_read_bytes (descending) and limit to top N
                sorted_metrics = sorted(
                    query_totals, key=lambda x: x.get("total_shared_hit_read_bytes", 0), reverse=True
                )[:limit]

                tracked_total = sum(q.get("total_shared_hit_read_bytes", 0) for q in sorted_metrics)
                other_total = sum(other)
                total_bytes = tracked_total + other_total

                queries_by_db[db_name] = {
                    "top_queries": sorted_metrics,
                    "other_shared_hit_read_bytes_hourly": other,
                    "summary": {
                        "queries_returned": len(sorted_metrics),
                        "total_shared_hit_read_bytes": total_bytes,
                        "total_shared_hit_read_bytes_tracked_queries": tracked_total,
                        "total_shared_hit_read_bytes_other": other_total,
                        "time_range_hours": hours,
                        "hourly_timestamps": timeline,
                        "limit": limit,
                    },
                }
        else:
            # Fallback for sub-hourly (not typically needed)
            pass

        return self.format_report_data(
            "K008",
            queries_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def generate_n001_wait_events_report(self, cluster: str = "local", node_name: str = "node-01",
                                         hours: int = 24) -> Dict[str, Any]:
        """
        Generate N001 Wait Events report with hourly breakdown grouped by wait_event_type and query_id.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            hours: Number of hours to analyze (default: 24)
            
        Returns:
            Dictionary containing wait events grouped by type and query_id with hourly occurrences
        """
        logger.info("Generating N001 Wait Events report...")

        # Get all databases
        databases = self.get_all_databases(cluster, node_name)
        
        if not databases:
            logger.warning("N001 - No databases found")

        # Build timeline
        now = int(time.time())
        end_s = self._floor_hour(now)
        start_s, timeline = self._build_timeline(end_s, hours, step_s=3600)

        wait_events_by_db = {}
        
        for db_name in databases:
            logger.info(f"N001: Processing database {db_name}...")
            
            # Query wait events from Prometheus
            # pgwatch_wait_events_total has labels: wait_event_type, wait_event, query_id, datname
            filters = [
                f'cluster="{cluster}"',
                f'node_name="{node_name}"',
                f'datname="{db_name}"'
            ]
            filter_str = '{' + ','.join(filters) + '}'
            
            # Get wait events data over the time range with hourly step
            metric_name = f'pgwatch_wait_events_total{filter_str}'
            
            try:
                result = self.query_range(metric_name, datetime.fromtimestamp(start_s), 
                                        datetime.fromtimestamp(end_s), step="3600s")
                
                if not result:
                    logger.warning(f"N001 - No wait events data for database {db_name}")
                    continue
                
                # Build timestamp to hour index map
                ts_to_hour = {ts: idx for idx, ts in enumerate(timeline)}
                
                # Process results to group by wait_event_type -> query_id with hourly breakdown
                wait_events_grouped = {}
                
                for series in result:
                    metric = series.get('metric', {})
                    wait_event_type = metric.get('wait_event_type', 'Unknown')
                    wait_event = metric.get('wait_event', 'Unknown')
                    query_id = metric.get('query_id', '0')
                    
                    # Get the values (timestamp, value pairs)
                    values = series.get('values', [])
                    
                    # Group by wait_event_type
                    if wait_event_type not in wait_events_grouped:
                        wait_events_grouped[wait_event_type] = {
                            'queries': {},
                            'total_occurrences': 0,
                            'unique_queries': 0
                        }
                    
                    # Add query_id under this wait_event_type
                    if query_id not in wait_events_grouped[wait_event_type]['queries']:
                        wait_events_grouped[wait_event_type]['queries'][query_id] = {
                            'occurrences': 0,
                            'hourly_occurrences': [0] * hours,
                            'wait_events': {}
                        }
                    
                    # Process hourly values
                    for timestamp, value in values:
                        try:
                            count = float(value)
                            if count == 0:
                                continue
                            
                            ts = int(timestamp)
                            if ts not in ts_to_hour:
                                continue
                            
                            hour_idx = ts_to_hour[ts]
                            
                            # Update hourly arrays for query only
                            wait_events_grouped[wait_event_type]['queries'][query_id]['hourly_occurrences'][hour_idx] += int(count)
                            
                            # Track individual wait events
                            if wait_event not in wait_events_grouped[wait_event_type]['queries'][query_id]['wait_events']:
                                wait_events_grouped[wait_event_type]['queries'][query_id]['wait_events'][wait_event] = {
                                    'occurrences': 0
                                }
                            wait_events_grouped[wait_event_type]['queries'][query_id]['wait_events'][wait_event]['occurrences'] += int(count)
                            
                        except (ValueError, TypeError):
                            continue
                
                # Calculate totals
                for wait_type in wait_events_grouped:
                    for query_id in wait_events_grouped[wait_type]['queries']:
                        query_data = wait_events_grouped[wait_type]['queries'][query_id]
                        query_data['occurrences'] = sum(query_data['hourly_occurrences'])
                    
                    # Calculate total_occurrences from all queries
                    wait_events_grouped[wait_type]['total_occurrences'] = sum(
                        q['occurrences'] for q in wait_events_grouped[wait_type]['queries'].values()
                    )
                
                # Skip databases with no wait events data
                if not wait_events_grouped:
                    logger.warning(f"N001 - No wait events data for database {db_name}")
                    continue
                
                # Count unique queries and convert to list
                for wait_type in wait_events_grouped:
                    wait_events_grouped[wait_type]['unique_queries'] = len(wait_events_grouped[wait_type]['queries'])
                    
                    queries_list = []
                    for query_id, data in wait_events_grouped[wait_type]['queries'].items():
                        queries_list.append({
                            'query_id': query_id,
                            'occurrences': data['occurrences'],
                            'hourly_occurrences': data['hourly_occurrences'],
                            'wait_events': data['wait_events']
                        })
                    # Sort by occurrences descending
                    queries_list.sort(key=lambda x: x['occurrences'], reverse=True)
                    wait_events_grouped[wait_type]['queries_list'] = queries_list
                    # Remove the dict version
                    del wait_events_grouped[wait_type]['queries']
                
                wait_events_by_db[db_name] = {
                    'wait_event_types': wait_events_grouped,
                    'summary': {
                        'time_range_hours': hours,
                        'start_time': datetime.fromtimestamp(start_s).isoformat(),
                        'end_time': datetime.fromtimestamp(end_s).isoformat(),
                        'wait_event_types_count': len(wait_events_grouped),
                        'total_occurrences': sum(wt['total_occurrences'] for wt in wait_events_grouped.values()),
                        'hourly_timestamps': timeline
                    }
                }
                
            except Exception as e:
                logger.error(f"Error querying wait events for database {db_name}: {e}")
                continue

        return self.format_report_data(
            "N001",
            wait_events_by_db,
            node_name,
            postgres_version=self._get_postgres_version_info(cluster, node_name),
        )

    def _get_pgss_metrics_data(self, cluster: str, node_name: str, start_time: datetime, end_time: datetime) -> List[
        Dict[str, Any]]:
        """
        Get pg_stat_statements metrics data between two time points.
        Adapted from the logic in monitoring_flask_backend/app.py get_pgss_metrics_csv().
        
        Args:
            cluster: Cluster name
            node_name: Node name  
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            List of query metrics with calculated differences
        """
        # Metric name mapping for cleaner output
        METRIC_NAME_MAPPING = {
            'calls': 'calls',
            'exec_time_total': 'total_time',
            'rows': 'rows',
            'shared_bytes_hit_total': 'shared_blks_hit',
            'shared_bytes_read_total': 'shared_blks_read',
            'shared_bytes_dirtied_total': 'shared_blks_dirtied',
            'shared_bytes_written_total': 'shared_blks_written',
            'block_read_total': 'blk_read_time',
            'block_write_total': 'blk_write_time'
        }

        # Build filters
        filters = [f'cluster="{cluster}"', f'node_name="{node_name}"']
        filter_str = '{' + ','.join(filters) + '}'

        # Get all pg_stat_statements metrics
        all_metrics = [
            'pgwatch_pg_stat_statements_calls',
            'pgwatch_pg_stat_statements_exec_time_total',
            'pgwatch_pg_stat_statements_rows',
            'pgwatch_pg_stat_statements_shared_bytes_hit_total',
            'pgwatch_pg_stat_statements_shared_bytes_read_total',
            'pgwatch_pg_stat_statements_shared_bytes_dirtied_total',
            'pgwatch_pg_stat_statements_shared_bytes_written_total',
            'pgwatch_pg_stat_statements_block_read_total',
            'pgwatch_pg_stat_statements_block_write_total'
        ]

        # Get metrics at start and end times
        start_data = []
        end_data = []

        for metric in all_metrics:
            metric_with_filters = f'{metric}{filter_str}'

            try:
                # Query metrics around start time - use instant queries at specific timestamps
                start_result = self.query_range(metric_with_filters, start_time - timedelta(minutes=1),
                                                start_time + timedelta(minutes=1))
                if start_result:
                    start_data.extend(start_result)

                # Query metrics around end time  
                end_result = self.query_range(metric_with_filters, end_time - timedelta(minutes=1),
                                              end_time + timedelta(minutes=1))
                if end_result:
                    end_data.extend(end_result)

            except Exception as e:
                logger.warning(f"Failed to query metric {metric}: {e}")
                continue

        # Process the data to calculate differences
        return self._process_pgss_data(start_data, end_data, start_time, end_time, METRIC_NAME_MAPPING)

    def query_range(self, query: str, start_time: datetime, end_time: datetime, step: str = "30s") -> List[
        Dict[str, Any]]:
        """
        Execute a range PromQL query.
        
        Args:
            query: PromQL query string
            start_time: Start time
            end_time: End time
            step: Query step interval
            
        Returns:
            List of query results
        """
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }

        try:
            response = requests.get(f"{self.base_url}/query_range", params=params, auth=self.auth)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    return result.get('data', {}).get('result', [])
            else:
                logger.error(f"Range query failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Range query error: {e}")

        return []

    def _process_pgss_data(self, start_data: List[Dict], end_data: List[Dict],
                           start_time: datetime, end_time: datetime,
                           metric_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Process pg_stat_statements data and calculate differences between start and end times.
        Adapted from the logic in monitoring_flask_backend/app.py process_pgss_data().
        """
        # Convert Prometheus data to dictionaries
        start_metrics = self._prometheus_to_dict(start_data, start_time)
        end_metrics = self._prometheus_to_dict(end_data, end_time)

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

            # Create result row
            row = {
                'queryid': query_id,
                'database': db_name,
                'user': user,
                'duration_seconds': actual_duration
            }

            # Numeric columns to calculate differences for (using original metric names)
            numeric_cols = list(metric_mapping.keys())

            # Calculate differences and rates
            for col in numeric_cols:
                start_val = start_metric.get(col, 0)
                end_val = end_metric.get(col, 0)
                diff = end_val - start_val

                # Use simplified display name
                display_name = metric_mapping[col]

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

        return result_rows

    def _prometheus_to_dict(self, prom_data: List[Dict], timestamp: datetime) -> Dict:
        """
        Convert Prometheus API response to dictionary keyed by query identifiers.
        Adapted from the logic in monitoring_flask_backend/app.py prometheus_to_dict().
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
            # Note: 'user' label may not exist in all metric configurations
            key = (
                metric.get('datname', ''),
                metric.get('queryid', ''),
                metric.get('user', metric.get('tag_user', '')),  # Fallback to tag_user or empty
                metric.get('instance', '')
            )

            # Initialize metric dict if not exists
            if key not in metrics_dict:
                metrics_dict[key] = {
                    'timestamp': datetime.fromtimestamp(float(closest_value[0])).isoformat(),
                }

            # Add metric value
            metric_name = metric.get('__name__', 'pgwatch_pg_stat_statements_calls')
            clean_name = metric_name.replace('pgwatch_pg_stat_statements_', '')

            try:
                metrics_dict[key][clean_name] = float(closest_value[1])
            except (ValueError, IndexError):
                metrics_dict[key][clean_name] = 0

        return metrics_dict

    def _floor_hour(self, ts: int) -> int:
        """
        Floor timestamp to the nearest hour, unless use_current_time is enabled.

        Args:
            ts: Unix timestamp in seconds

        Returns:
            Floored timestamp (or original timestamp if use_current_time is True)
        """
        if self.use_current_time:
            return ts
        return (ts // 3600) * 3600

    def _build_timeline(self, end_s: int, hours: int = 24, step_s: int = 3600) -> Tuple[int, List[int]]:
        """
        Build a timeline of hourly timestamps.
        
        Args:
            end_s: End timestamp (floored to hour)
            hours: Number of hours to cover (default: 24)
            step_s: Step size in seconds (default: 3600 = 1 hour)
            
        Returns:
            Tuple of (start_timestamp, list of timestamps)
        """
        start_s = end_s - (hours - 1) * step_s
        return start_s, [start_s + i * step_s for i in range(hours)]

    def _build_qid_regex(self, qids: List[str]) -> str:
        """
        Build a PromQL regex pattern for queryid matching.
        
        Args:
            qids: List of query IDs
            
        Returns:
            PromQL regex pattern
        """
        # queryid is integer-like (can be negative). DO NOT escape '-' for PromQL strings.
        for q in qids:
            if not re.fullmatch(r"-?\d+", q):
                raise ValueError(f"Unexpected queryid: {q}")
        return "^(?:" + "|".join(qids) + ")$"

    def _to_series_map(self, result: List[Dict]) -> Dict[str, Dict[int, float]]:
        """
        Convert Prometheus query_range result to a map of series.
        
        Args:
            result: Prometheus query_range result
            
        Returns:
            Dict mapping queryid to dict of timestamp -> value
        """
        out = {}
        for s in result:
            qid = (s.get("metric") or {}).get("queryid", "__single__")
            pts = {int(ts): float(v) for ts, v in s.get("values", [])}
            out[qid] = pts
        return out

    def _densify(self, series_pts: Dict[str, Dict[int, float]], qids: List[str], 
                 timeline: List[int], fill: float = 0.0) -> Dict[str, List[float]]:
        """
        Densify sparse series data to have values for all timeline points.
        
        Args:
            series_pts: Map of queryid to timestamp -> value
            qids: List of query IDs to densify
            timeline: List of timestamps
            fill: Fill value for missing data points (default: 0.0)
            
        Returns:
            Dict mapping queryid to list of values aligned to timeline
        """
        return {
            qid: [series_pts.get(qid, {}).get(ts, fill) for ts in timeline] 
            for qid in qids
        }

    def _get_hourly_topk_pgss_data_multi(
        self,
        cluster: str,
        node_name: str,
        db_name: str,
        metric_names: Sequence[str],
        hours: int = 24,
        step_s: int = 3600,
        k: int = 3,
    ) -> Tuple[Dict[str, List[float]], List[float], List[int]]:
        """
        Generalization of `_get_hourly_topk_pgss_data` that ranks and returns per-hour series by the
        sum of one or more pg_stat_statements Prometheus metrics.
        """
        metric_names = [m for m in metric_names if m]
        if not metric_names:
            raise ValueError("metric_names must contain at least one metric name")

        now = int(time.time())
        end_s = self._floor_hour(now)
        start_s, timeline = self._build_timeline(end_s, hours, step_s)

        filters = [f'cluster="{cluster}"', f'node_name="{node_name}"', f'datname="{db_name}"']
        filter_str = '{' + ','.join(filters) + '}'
        step_str = f"{step_s}s"

        def _sum_by_qid_increase(metric: str, fstr: str) -> str:
            return f"sum by (queryid) (increase({metric}{fstr}[1h]))"

        def _sum_increase(metric: str, fstr: str) -> str:
            return f"sum(increase({metric}{fstr}[1h]))"

        # Find union of queryids that ever appear in hourly top-k by the sum of metrics.
        topk_expr = " + ".join(_sum_by_qid_increase(m, filter_str) for m in metric_names)
        q_topk = f"topk({k}, ({topk_expr}))"
        topk_result = self.query_range(
            q_topk, datetime.fromtimestamp(start_s), datetime.fromtimestamp(end_s), step=step_str
        )
        union = sorted(
            {
                (s.get("metric") or {}).get("queryid")
                for s in topk_result
                if (s.get("metric") or {}).get("queryid") is not None
            }
        )

        # Total per hour (for "other" calculation)
        total_expr = " + ".join(_sum_increase(m, filter_str) for m in metric_names)
        q_total = f"({total_expr})"
        total_result = self.query_range(
            q_total, datetime.fromtimestamp(start_s), datetime.fromtimestamp(end_s), step=step_str
        )
        total_map = self._to_series_map(total_result).get("__single__", {})
        total = [total_map.get(ts, 0.0) for ts in timeline]

        if not union:
            return {}, total[:], timeline

        # Hourly series for all union queryids (densified to N points each)
        qid_re = self._build_qid_regex(union)
        union_filters = filters + [f'queryid=~"{qid_re}"']
        union_filter_str = '{' + ','.join(union_filters) + '}'
        union_expr = " + ".join(_sum_by_qid_increase(m, union_filter_str) for m in metric_names)
        q_union = f"({union_expr})"
        union_result = self.query_range(
            q_union, datetime.fromtimestamp(start_s), datetime.fromtimestamp(end_s), step=step_str
        )
        union_pts = self._to_series_map(union_result)
        per_query = self._densify(union_pts, union, timeline, fill=0.0)

        # Calculate other = total - sum(union)
        other: List[float] = []
        neg_examples: List[Tuple[int, float, float, float]] = []
        for i in range(hours):
            union_sum = sum(per_query[qid][i] for qid in union)
            o_raw = total[i] - union_sum
            if o_raw < 0:
                # Keep small float noise quiet, but surface meaningful negatives.
                if o_raw < -1e-6 and len(neg_examples) < 5:
                    neg_examples.append((timeline[i], o_raw, total[i], union_sum))
                o_raw = 0.0
            other.append(o_raw)

        if neg_examples:
            min_neg = min(v[1] for v in neg_examples)
            logger.warning(
                "Hourly topk: negative 'other' clamped to 0 "
                f"(cluster={cluster}, node={node_name}, db={db_name}, metrics={list(metric_names)}, "
                f"hours={hours}, step_s={step_s}, k={k}, min_other={min_neg:.6g}, "
                f"examples={neg_examples})"
            )

        return per_query, other, timeline

    def _get_hourly_topk_pgss_data(self, cluster: str, node_name: str, db_name: str,
                                   metric_name: str = "pgwatch_pg_stat_statements_calls",
                                   hours: int = 24, step_s: int = 3600, 
                                   k: int = 3) -> Tuple[Dict[str, List[float]], List[float], List[int]]:
        """
        Get hourly topk pg_stat_statements data for a specific database and metric.
        
        This method finds queries that appear in top-k for any hour within the time range,
        then returns per-hour data for those queries plus an "other" category.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            db_name: Database name
            metric_name: Prometheus metric name (default: pgwatch_pg_stat_statements_calls)
            hours: Number of hours to look back (default: 24)
            step_s: Step size in seconds (default: 3600 = 1 hour)
            k: Number of top queries per hour (default: 3)
            
        Returns:
            Tuple of (per_query_dict, other_list, timeline)
            - per_query_dict: Dict mapping queryid to list of hourly values
            - other_list: List of hourly values for queries not in top-k
            - timeline: List of timestamps for the hourly data points
        """
        return self._get_hourly_topk_pgss_data_multi(
            cluster=cluster,
            node_name=node_name,
            db_name=db_name,
            metric_names=[metric_name],
            hours=hours,
            step_s=step_s,
            k=k,
        )

    def _get_hourly_topk_pgss_data_sum2(
        self,
        cluster: str,
        node_name: str,
        db_name: str,
        metric_name_a: str,
        metric_name_b: str,
        hours: int = 24,
        step_s: int = 3600,
        k: int = 3,
    ) -> Tuple[Dict[str, List[float]], List[float], List[int]]:
        """
        Like `_get_hourly_topk_pgss_data`, but ranks by the sum of two metrics per queryid, per hour:

          sum by(queryid)(increase(A[1h])) + sum by(queryid)(increase(B[1h]))

        This avoids a correctness pitfall where "union(topk by A, topk by B)" can miss a query that is
        not top-k in either A or B individually, but is top-k by (A+B).
        """
        return self._get_hourly_topk_pgss_data_multi(
            cluster=cluster,
            node_name=node_name,
            db_name=db_name,
            metric_names=[metric_name_a, metric_name_b],
            hours=hours,
            step_s=step_s,
            k=k,
        )

    def format_bytes(self, bytes_value: float) -> str:
        """Format bytes value for human readable display."""
        if bytes_value == 0:
            return "0 B"

        # Use IEC binary prefixes because we divide by 1024.
        units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
        unit_index = 0
        value = float(bytes_value)

        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1

        if value >= 100:
            return f"{value:.0f} {units[unit_index]}"
        elif value >= 10:
            return f"{value:.1f} {units[unit_index]}"
        else:
            return f"{value:.2f} {units[unit_index]}"

    def format_epoch_timestamp(self, epoch_value: float) -> str | None:
        """Format epoch seconds as a UTC timestamptz string (ISO-8601, like `timestamptz` in reports)."""
        try:
            v = float(epoch_value or 0)
        except (TypeError, ValueError):
            return None

        if v <= 0:
            return None

        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    def format_report_data(self, check_id: str, data: Dict[str, Any], host: str = "target-database", 
                          all_hosts: Dict[str, List[str]] = None,
                          postgres_version: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Format data to match template structure.
        
        Args:
            check_id: The check identifier
            data: The data to format (can be a dict with node keys if combining multiple nodes)
            host: Primary host identifier (used if all_hosts not provided)
            all_hosts: Optional dict with 'primary' and 'standbys' keys for multi-node reports
            postgres_version: Optional Postgres version info to include at report level
            
        Returns:
            Dictionary formatted for templates
        """
        now = datetime.now(timezone.utc)

        # If all_hosts is provided, use it; otherwise use the single host as primary
        if all_hosts:
            hosts = all_hosts
        else:
            hosts = {
                "primary": host,
                "standbys": [],
            }

        # Handle both single-node and multi-node data structures
        if isinstance(data, dict) and any(isinstance(v, dict) and 'data' in v for v in data.values()):
            # Multi-node structure: data is already in {node_name: {"data": ...}} format
            # postgres_version should already be embedded per-node; warn if passed here
            if postgres_version:
                logger.warning(f"postgres_version parameter ignored for multi-node data in {check_id}")
            results = data
        else:
            # Single-node structure: wrap data in host key
            node_result = {"data": data}
            if postgres_version:
                node_result["postgres_version"] = postgres_version
            results = {host: node_result}

        template_data = {
            "version": self._build_metadata.get("version"),
            "build_ts": self._build_metadata.get("build_ts"),
            "generation_mode": "full",
            "checkId": check_id,
            "checkTitle": self.get_check_title(check_id),
            "timestamptz": now.isoformat(),
            "nodes": hosts,
            "results": results
        }

        return template_data

    def filter_a003_settings(self, a003_report: Dict[str, Any], setting_names: List[str]) -> Dict[str, Any]:
        """
        Filter A003 settings data to include only specified settings.

        Args:
            a003_report: Full A003 report containing all settings
            setting_names: List of setting names to include

        Returns:
            Filtered settings dictionary
        """
        filtered = {}
        # Handle both single-node and multi-node A003 report structures
        results = a003_report.get('results', {})
        for node_name, node_data in results.items():
            data = node_data.get('data', {})
            for setting_name, setting_info in data.items():
                if setting_name in setting_names:
                    filtered[setting_name] = setting_info
        return filtered

    def extract_postgres_version_from_a003(self, a003_report: Dict[str, Any], node_name: str = None) -> Dict[str, str]:
        """
        Extract PostgreSQL version info from A003 report settings data.

        Derives version from server_version and server_version_num settings
        which are part of the A003 settings data.

        Args:
            a003_report: Full A003 report
            node_name: Optional specific node name. If None, uses first available node.

        Returns:
            Dictionary with postgres version info (version, server_version_num, server_major_ver, server_minor_ver)
        """
        results = a003_report.get('results', {})
        if not results:
            return {}

        # Get the node data
        if node_name and node_name in results:
            node_data = results[node_name]
        else:
            node_data = next(iter(results.values()), {})

        # First check if postgres_version is already in the node result
        if node_data.get('postgres_version'):
            return node_data['postgres_version']

        # Otherwise, extract from settings data (server_version, server_version_num)
        data = node_data.get('data', {})
        version_str = None
        version_num = None

        # Look for server_version and server_version_num in settings
        if 'server_version' in data:
            version_str = data['server_version'].get('setting', '')
        if 'server_version_num' in data:
            version_num = data['server_version_num'].get('setting', '')

        if not version_str and not version_num:
            return {}

        # Parse version numbers
        major_ver = ""
        minor_ver = ""
        if version_num and len(version_num) >= 6:
            try:
                num = int(version_num)
                major_ver = str(num // 10000)
                minor_ver = str(num % 10000)
            except ValueError:
                pass

        return {
            "version": version_str or "",
            "server_version_num": version_num or "",
            "server_major_ver": major_ver,
            "server_minor_ver": minor_ver
        }

    def generate_d004_from_a003(self, a003_report: Dict[str, Any], cluster: str = "local",
                                 node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate D004 report by filtering A003 data for pg_stat_statements settings.

        Args:
            a003_report: Full A003 report containing all settings
            cluster: Cluster name (for status checks)
            node_name: Node name

        Returns:
            D004 report dictionary
        """
        print("Generating D004 from A003 data...")

        # Filter A003 settings for D004-relevant settings
        pgstat_data = self.filter_a003_settings(a003_report, self.D004_SETTINGS)

        # Check extension status (still needs direct queries)
        kcache_status = self._check_pg_stat_kcache_status(cluster, node_name)
        pgss_status = self._check_pg_stat_statements_status(cluster, node_name)

        # Extract postgres version from A003
        postgres_version = self.extract_postgres_version_from_a003(a003_report, node_name)

        return self.format_report_data(
            "D004",
            {
                "settings": pgstat_data,
                "pg_stat_statements_status": pgss_status,
                "pg_stat_kcache_status": kcache_status,
            },
            node_name,
            postgres_version=postgres_version,
        )

    def generate_f001_from_a003(self, a003_report: Dict[str, Any], node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate F001 report by filtering A003 data for autovacuum settings.

        Args:
            a003_report: Full A003 report containing all settings
            node_name: Node name

        Returns:
            F001 report dictionary
        """
        print("Generating F001 from A003 data...")

        # Filter A003 settings for F001-relevant settings
        autovacuum_data = self.filter_a003_settings(a003_report, self.F001_SETTINGS)

        # Extract postgres version from A003
        postgres_version = self.extract_postgres_version_from_a003(a003_report, node_name)

        return self.format_report_data("F001", autovacuum_data, node_name, postgres_version=postgres_version)

    def generate_g001_from_a003(self, a003_report: Dict[str, Any], node_name: str = "node-01") -> Dict[str, Any]:
        """
        Generate G001 report by filtering A003 data for memory settings.

        Args:
            a003_report: Full A003 report containing all settings
            node_name: Node name

        Returns:
            G001 report dictionary with memory analysis
        """
        print("Generating G001 from A003 data...")

        # Filter A003 settings for G001-relevant settings
        memory_data = self.filter_a003_settings(a003_report, self.G001_SETTINGS)

        # Calculate memory analysis
        memory_analysis = self._analyze_memory_settings(memory_data)

        # Extract postgres version from A003
        postgres_version = self.extract_postgres_version_from_a003(a003_report, node_name)

        return self.format_report_data(
            "G001",
            {
                "settings": memory_data,
                "analysis": memory_analysis,
            },
            node_name,
            postgres_version=postgres_version,
        )

    def get_check_title(self, check_id: str) -> str:
        """
        Get the human-readable title for a check ID.

        Args:
            check_id: The check identifier (e.g., "H004")

        Returns:
            Human-readable title for the check
        """
        # Mapping based on postgres-checkup README
        # https://gitlab.com/postgres-ai/postgres-checkup
        check_titles = {
            "A001": "System information",
            "A002": "Postgres major version",
            "A003": "Postgres settings",
            "A004": "Cluster information",
            "A005": "Extensions",
            "A006": "Postgres setting deviations",
            "A007": "Altered settings",
            "A008": "Disk usage and file system type",
            "A010": "Data checksums, wal_log_hints",
            "A011": "Connection pooling. pgbouncer",
            "A012": "Anti-crash checks",
            "A013": "Postgres minor version",
            "B001": "SLO/SLA, RPO, RTO",
            "B002": "File system, mount flags",
            "B003": "Full backups / incremental",
            "B004": "WAL archiving",
            "B005": "Restore checks, monitoring, alerting",
            "C001": "SLO/SLA",
            "C002": "Sync/async, Streaming / wal transfer; logical decoding",
            "C003": "SPOFs; standby with traffic",
            "C004": "Failover",
            "C005": "Switchover",
            "C006": "Delayed replica",
            "C007": "Replication slots. Lags. Standby feedbacks",
            "D001": "Logging settings",
            "D002": "Useful Linux tools",
            "D003": "List of monitoring metrics",
            "D004": "pg_stat_statements and pg_stat_kcache settings",
            "D005": "track_io_timing, auto_explain",
            "D006": "Recommended DBA toolsets",
            "D007": "Postgres-specific tools for troubleshooting",
            "E001": "WAL/checkpoint settings, IO",
            "E002": "Checkpoints, bgwriter, IO",
            "F001": "Autovacuum: current settings",
            "F002": "Autovacuum: transaction ID wraparound check",
            "F003": "Autovacuum: dead tuples",
            "F004": "Autovacuum: heap bloat (estimated)",
            "F005": "Autovacuum: index bloat (estimated)",
            "F006": "Precise heap bloat analysis",
            "F007": "Precise index bloat analysis",
            "F008": "Autovacuum: resource usage",
            "G001": "Memory-related settings",
            "G002": "Connections and current activity",
            "G003": "Timeouts, locks, deadlocks",
            "G004": "Query planner",
            "G005": "I/O settings",
            "G006": "Default_statistics_target",
            "H001": "Invalid indexes",
            "H002": "Unused indexes",
            "H003": "Non-indexed foreign keys",
            "H004": "Redundant indexes",
            "J001": "Capacity planning",
            "K001": "Globally aggregated query metrics",
            "K002": "Workload type",
            "K003": "Top queries by total time (total_exec_time + total_plan_time)",
            "K004": "Top queries by temp bytes written",
            "K005": "Top queries by WAL generation",
            "K006": "Top queries by shared blocks read",
            "K007": "Top queries by shared blocks hit",
            "K008": "Top queries by shared blocks hit+read",
            "L001": "Table sizes",
            "M001": "Top queries by mean execution time",
            "M002": "Top queries by rows (I/O intensity)",
            "M003": "Top queries by I/O time",
            "N001": "Wait events grouped by type and query",
            "L002": "Data types being used",
            "L003": "Integer out-of-range risks in PKs",
            "L004": "Tables without PK/UK",
        }
        return check_titles.get(check_id, f"Check {check_id}")

    def get_setting_unit(self, setting_name: str) -> str:
        """Get the unit for a PostgreSQL setting."""
        units = {
            'max_connections': 'connections',
            'shared_buffers': '8kB',
            'effective_cache_size': '8kB',
            'work_mem': 'kB',
            'maintenance_work_mem': 'kB',
            'checkpoint_completion_target': '',
            'wal_buffers': '8kB',
            'default_statistics_target': '',
            'random_page_cost': '',
            'effective_io_concurrency': '',
            'autovacuum_max_workers': 'workers',
            'autovacuum_naptime': 's',
            'log_min_duration_statement': 'ms',
            'idle_in_transaction_session_timeout': 'ms',
            'lock_timeout': 'ms',
            'statement_timeout': 'ms',
        }
        return units.get(setting_name, '')

    def get_setting_category(self, setting_name: str) -> str:
        """Get the category for a PostgreSQL setting."""
        categories = {
            'max_connections': 'Connections and Authentication',
            'shared_buffers': 'Memory',
            'effective_cache_size': 'Memory',
            'work_mem': 'Memory',
            'maintenance_work_mem': 'Memory',
            'checkpoint_completion_target': 'Write-Ahead Logging',
            'wal_buffers': 'Write-Ahead Logging',
            'default_statistics_target': 'Query Planning',
            'random_page_cost': 'Query Planning',
            'effective_io_concurrency': 'Asynchronous Behavior',
            'autovacuum_max_workers': 'Autovacuum',
            'autovacuum_naptime': 'Autovacuum',
            'log_min_duration_statement': 'Logging',
            'idle_in_transaction_session_timeout': 'Client Connection Defaults',
            'lock_timeout': 'Client Connection Defaults',
            'statement_timeout': 'Client Connection Defaults',
        }
        return categories.get(setting_name, 'Other')

    def format_setting_value(self, setting_name: str, value: str, unit: str = "") -> str:
        """Format a setting value for display."""
        try:
            # If we have a unit from the metric, use it
            if unit:
                if unit == "8kB":
                    val = int(value) * 8
                    if val >= 1024 and val % 1024 == 0:
                        return f"{val // 1024} MiB"
                    else:
                        return f"{val} KiB"
                elif unit == "ms":
                    val = int(value)
                    if val >= 1000 and val % 1000 == 0:
                        return f"{val // 1000} s"
                    else:
                        return f"{val} ms"
                elif unit == "s":
                    return f"{value} s"
                elif unit == "min":
                    return f"{value} min"
                elif unit == "connections":
                    return f"{value} connections"
                elif unit == "workers":
                    return f"{value} workers"
                else:
                    return f"{value} {unit}"

            # Fallback to setting name based formatting
            if setting_name in ['shared_buffers', 'effective_cache_size', 'work_mem', 'maintenance_work_mem',
                                'autovacuum_work_mem', 'logical_decoding_work_mem', 'temp_buffers', 'wal_buffers']:
                val = int(value)
                if val >= 1024:
                    return f"{val // 1024} MiB"
                else:
                    return f"{val} KiB"
            elif setting_name in ['log_min_duration_statement', 'idle_in_transaction_session_timeout', 'lock_timeout',
                                  'statement_timeout', 'autovacuum_vacuum_cost_delay', 'vacuum_cost_delay']:
                val = int(value)
                if val >= 1000:
                    return f"{val // 1000} s"
                else:
                    return f"{val} ms"
            elif setting_name in ['autovacuum_naptime']:
                val = int(value)
                if val >= 60:
                    return f"{val // 60} min"
                else:
                    return f"{val} s"
            elif setting_name in ['autovacuum_max_workers']:
                return f"{value} workers"
            elif setting_name in ['pg_stat_statements.max']:
                return f"{value} statements"
            elif setting_name in ['max_wal_size', 'min_wal_size']:
                val = int(value)
                if val >= 1024:
                    return f"{val // 1024} GiB"
                else:
                    return f"{val} MiB"
            elif setting_name in ['checkpoint_completion_target']:
                return f"{float(value):.2f}"
            elif setting_name in ['hash_mem_multiplier']:
                return f"{float(value):.1f}"
            elif setting_name in ['max_connections', 'max_prepared_transactions', 'max_locks_per_transaction',
                                  'max_pred_locks_per_transaction', 'max_pred_locks_per_relation',
                                  'max_pred_locks_per_page', 'max_files_per_process']:
                return f"{value} connections" if "connections" in setting_name else f"{value}"
            elif setting_name in ['max_stack_depth']:
                val = int(value)
                if val >= 1024:
                    return f"{val // 1024} MiB"
                else:
                    return f"{val} KiB"
            elif setting_name in ['autovacuum_analyze_scale_factor', 'autovacuum_vacuum_scale_factor',
                                  'autovacuum_vacuum_insert_scale_factor']:
                return f"{float(value) * 100:.1f}%"
            elif setting_name in ['autovacuum', 'track_activities', 'track_counts', 'track_functions',
                                  'track_io_timing', 'track_wal_io_timing', 'pg_stat_statements.track_utility',
                                  'pg_stat_statements.save', 'pg_stat_statements.track_planning']:
                return "on" if value.lower() in ['on', 'true', '1'] else "off"
            elif setting_name in ['huge_pages']:
                return value  # on/off/try
            else:
                return str(value)
        except (ValueError, TypeError):
            return str(value)

    def get_cluster_metric_unit(self, metric_name: str) -> str:
        """Get the unit for a cluster metric."""
        units = {
            'active_connections': 'connections',
            'idle_connections': 'connections',
            'total_connections': 'connections',
            'database_size': 'bytes',
            'cache_hit_ratio': '%',
            'transactions_per_sec': 'tps',
            'checkpoints_per_sec': 'checkpoints/s',
            'deadlocks': 'count',
            'temp_files': 'files',
            'temp_bytes': 'bytes',
        }
        return units.get(metric_name, '')

    def get_cluster_metric_description(self, metric_name: str) -> str:
        """Get the description for a cluster metric."""
        descriptions = {
            'active_connections': 'Number of active connections',
            'idle_connections': 'Number of idle connections',
            'total_connections': 'Total number of connections',
            'database_size': 'Total database size in bytes',
            'cache_hit_ratio': 'Cache hit ratio percentage',
            'transactions_per_sec': 'Transactions per second',
            'checkpoints_per_sec': 'Checkpoints per second',
            'deadlocks': 'Number of deadlocks',
            'temp_files': 'Number of temporary files',
            'temp_bytes': 'Size of temporary files in bytes',
        }
        return descriptions.get(metric_name, '')

    def generate_all_reports(self, cluster: str = "local", node_name: str = None, combine_nodes: bool = True) -> Dict[str, Any]:
        """
        Generate all reports.
        
        Args:
            cluster: Cluster name
            node_name: Node name (if None and combine_nodes=True, will query all nodes)
            combine_nodes: If True, combine primary and replica reports into single report
            
        Returns:
            Dictionary containing all reports
        """
        reports = {}

        # Determine which nodes to process
        if combine_nodes and node_name is None:
            # Get all nodes and combine them
            all_nodes = self.get_all_nodes(cluster)
            nodes_to_process = []
            if all_nodes["primary"]:
                nodes_to_process.append(all_nodes["primary"])
            nodes_to_process.extend(all_nodes["standbys"])
            
            # If no nodes found, fall back to default
            if not nodes_to_process:
                logger.warning(f"No nodes found in cluster '{cluster}', using default 'node-01'")
                nodes_to_process = ["node-01"]
                all_nodes = {"primary": "node-01", "standbys": []}
            else:
                logger.info(f"Combining reports from nodes: {nodes_to_process}")
        else:
            # Use single node (backward compatibility)
            if node_name is None:
                node_name = "node-01"
            nodes_to_process = [node_name]
            all_nodes = {"primary": node_name, "standbys": []}

        # Reports that don't depend on A003 (generate first)
        independent_report_types = [
            ('A002', self.generate_a002_version_report),
            ('A003', self.generate_a003_settings_report),
            ('A004', self.generate_a004_cluster_report),
            ('A007', self.generate_a007_altered_settings_report),
            ('F004', self.generate_f004_heap_bloat_report),
            ('F005', self.generate_f005_btree_bloat_report),
            ('H001', self.generate_h001_invalid_indexes_report),
            ('H002', self.generate_h002_unused_indexes_report),
            ('H004', self.generate_h004_redundant_indexes_report),
            ('K001', self.generate_k001_query_calls_report),
            ('K003', self.generate_k003_top_queries_report),
            ('K004', self.generate_k004_temp_bytes_report),
            ('K005', self.generate_k005_wal_bytes_report),
            ('K006', self.generate_k006_shared_read_report),
            ('K007', self.generate_k007_shared_hit_report),
            ('K008', self.generate_k008_shared_hit_read_report),
            ('M001', self.generate_m001_mean_time_report),
            ('M002', self.generate_m002_rows_report),
            ('M003', self.generate_m003_io_time_report),
            ('N001', self.generate_n001_wait_events_report),
        ]

        for check_id, report_func in independent_report_types:
            # Determine if this report needs hourly parameters
            pgss_hourly_reports = ['K001', 'K003', 'K004', 'K005', 'K006', 'K007', 'K008', 'M001', 'M002', 'M003']
            wait_events_reports = ['N001']
            report_kwargs = {}
            if check_id in pgss_hourly_reports:
                report_kwargs['time_range_minutes'] = 1440  # 24 hours
            elif check_id in wait_events_reports:
                report_kwargs['hours'] = 24  # 24 hours
            
            if len(nodes_to_process) == 1:
                # Single node - generate report normally
                reports[check_id] = report_func(cluster, nodes_to_process[0], **report_kwargs)
            else:
                # Multiple nodes - combine reports
                combined_results = {}
                for node in nodes_to_process:
                    logger.info(f"Generating {check_id} report for node {node}...")
                    node_report = report_func(cluster, node, **report_kwargs)
                    # Extract the data from the node report
                    if 'results' in node_report and node in node_report['results']:
                        combined_results[node] = node_report['results'][node]
                    
                    # Free node report memory immediately
                    del node_report
                
                # Create combined report with all nodes
                reports[check_id] = self.format_report_data(
                    check_id,
                    combined_results,
                    all_nodes["primary"] if all_nodes["primary"] else nodes_to_process[0],
                    all_nodes
                )
                
                # Free combined results after creating report
                del combined_results
            
            # Periodic garbage collection during report generation
            if len(reports) % 5 == 0:
                gc.collect()

        # Generate D004, F001, G001 from A003 data (if A003 was generated successfully)
        a003_report = reports.get('A003')
        if a003_report:
            # Reports derived from A003
            a003_derived_reports = [
                ('D004', lambda c, n: self.generate_d004_from_a003(a003_report, c, n)),
                ('F001', lambda c, n: self.generate_f001_from_a003(a003_report, n)),
                ('G001', lambda c, n: self.generate_g001_from_a003(a003_report, n)),
            ]

            for check_id, report_func in a003_derived_reports:
                if len(nodes_to_process) == 1:
                    reports[check_id] = report_func(cluster, nodes_to_process[0])
                else:
                    # For multi-node, use the first node as reference
                    # (A003 data already contains all nodes)
                    combined_results = {}
                    for node in nodes_to_process:
                        print(f"Generating {check_id} report for node {node} from A003...")
                        node_report = report_func(cluster, node)
                        if 'results' in node_report and node in node_report['results']:
                            combined_results[node] = node_report['results'][node]

                    reports[check_id] = self.format_report_data(
                        check_id,
                        combined_results,
                        all_nodes["primary"] if all_nodes["primary"] else nodes_to_process[0],
                        all_nodes
                    )
        else:
            # Fallback to direct generation if A003 failed
            print("Warning: A003 report not available, generating D004/F001/G001 directly")
            fallback_report_types = [
                ('D004', self.generate_d004_pgstat_settings_report),
                ('F001', self.generate_f001_autovacuum_settings_report),
                ('G001', self.generate_g001_memory_settings_report),
            ]
            for check_id, report_func in fallback_report_types:
                if len(nodes_to_process) == 1:
                    reports[check_id] = report_func(cluster, nodes_to_process[0])
                else:
                    combined_results = {}
                    for node in nodes_to_process:
                        print(f"Generating {check_id} report for node {node}...")
                        node_report = report_func(cluster, node)
                        if 'results' in node_report and node in node_report['results']:
                            combined_results[node] = node_report['results'][node]

                    reports[check_id] = self.format_report_data(
                        check_id,
                        combined_results,
                        all_nodes["primary"] if all_nodes["primary"] else nodes_to_process[0],
                        all_nodes
                    )

        return reports

    def generate_queries_json(self, query_text_limit: int = 1000) -> Dict[str, List[str]]:
        """
        DEPRECATED: This method is no longer used.
        Query information is now only included in individual query_{queryid}.json files.
        
        Generate JSON with queryid lists per database.

        Args:
            query_text_limit: Not used anymore, kept for backward compatibility
        
        Returns:
            Dictionary with database names as keys, containing lists of queryids
        """
        logger.warning("DEPRECATED: generate_queries_json is no longer used")
        queries_with_text = self.get_queryid_queries_from_sink(query_text_limit)
        
        # Convert from {db: {queryid: text}} to {db: [queryid, ...]}
        queries_only = {}
        for db_name, queries in queries_with_text.items():
            queries_only[db_name] = list(queries.keys())
        
        return queries_only

    def extract_queryids_from_reports(self, reports: Dict[str, Any]) -> Dict[str, set]:
        """
        Extract all unique queryids from the hourly reports (K001-K007, M001-M003, N001).
        
        Args:
            reports: Dictionary of generated reports keyed by check_id
            
        Returns:
            Dictionary mapping database names to sets of queryids
        """
        queryids_by_db: Dict[str, set] = {}
        
        def extract_from_query_metrics(
            container: Dict,
            target_key: str = 'query_metrics',
            id_field: str = 'queryid'
        ):
            """
            Helper to extract queryids from a container that may have nested structure.

            Notes:
            - Different reports use different list keys for per-query items:
              - K001 uses 'query_metrics'
              - K003-K007 and M001-M003 use 'top_queries'
            - We only keep queryids when we can associate them with a db_name, because
              per-query file generation later needs (cluster, node, db, queryid) to query Prometheus.
            """
            if not isinstance(container, dict):
                return
            
            # Direct: container has query_metrics
            if target_key in container:
                for query in container.get(target_key, []):
                    qid = query.get(id_field)
                    if qid and str(qid) != '0':
                        # Try to find db_name from context or use a placeholder
                        yield str(qid), None
            
            # Check for 'data' wrapper: container -> data -> db_name -> query_metrics
            if 'data' in container and isinstance(container['data'], dict):
                for db_name, db_data in container['data'].items():
                    if isinstance(db_data, dict) and target_key in db_data:
                        for query in db_data.get(target_key, []):
                            qid = query.get(id_field)
                            if qid and str(qid) != '0':
                                yield str(qid), db_name
            
            # Direct db_name -> query_metrics (no data wrapper)
            for key, value in container.items():
                if key == 'data':
                    continue
                if isinstance(value, dict) and target_key in value:
                    for query in value.get(target_key, []):
                        qid = query.get(id_field)
                        if qid and str(qid) != '0':
                            yield str(qid), key
        
        # Reports with queryid field in query_metrics list
        pgss_reports = ['K001', 'K003', 'K004', 'K005', 'K006', 'K007', 'K008', 'M001', 'M002', 'M003']
        
        for report_id in pgss_reports:
            if report_id not in reports:
                continue
            
            report = reports[report_id]
            results = report.get('results', {})
            
            # Handle multi-node structure: results -> node_name -> data -> db_name -> query_metrics
            for node_key, node_data in results.items():
                if isinstance(node_data, dict):
                    # K001 uses 'query_metrics', while most other hourly/topk reports use 'top_queries'.
                    for list_key in ('query_metrics', 'top_queries'):
                        for queryid, db_name in extract_from_query_metrics(node_data, target_key=list_key):
                            if db_name:
                                if db_name not in queryids_by_db:
                                    queryids_by_db[db_name] = set()
                                queryids_by_db[db_name].add(queryid)
        
        # N001 Wait Events report - has query_id in queries_list under wait_event_types
        if 'N001' in reports:
            report = reports['N001']
            results = report.get('results', {})
            
            for node_key, node_data in results.items():
                if not isinstance(node_data, dict):
                    continue
                
                # Check for 'data' wrapper
                data_container = node_data.get('data', node_data)
                
                for db_name, db_data in data_container.items():
                    if not isinstance(db_data, dict):
                        continue
                    
                    wait_types = db_data.get('wait_event_types', {})
                    if not wait_types:
                        continue
                    
                    if db_name not in queryids_by_db:
                        queryids_by_db[db_name] = set()
                    
                    for wait_type, wait_data in wait_types.items():
                        for query in wait_data.get('queries_list', []):
                            query_id = query.get('query_id')
                            if query_id and str(query_id) != '0':
                                queryids_by_db[db_name].add(str(query_id))
        
        # Log summary
        total_queryids = sum(len(qids) for qids in queryids_by_db.values())
        logger.info(f"Extracted {total_queryids} unique queryids from hourly reports across {len(queryids_by_db)} database(s)")
        
        return queryids_by_db

    def get_query_metrics_from_prometheus(self, cluster: str, node_name: str, db_name: str,
                                          queryid: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get all pg_stat_statements metrics for a specific query directly from Prometheus.
        Fetches daily totals for all metrics shown on Dashboard 3 (Single queryid analysis).
        
        Args:
            cluster: Cluster name
            node_name: Node name
            db_name: Database name
            queryid: Query ID
            hours: Number of hours to aggregate (default: 24 for daily totals)
            
        Returns:
            Dictionary of metrics with daily totals
        """
        metrics = {}
        
        # Build filters for this specific query
        filters = [
            f'cluster="{cluster}"',
            f'node_name="{node_name}"',
            f'datname="{db_name}"',
            f'queryid="{queryid}"'
        ]
        filter_str = '{' + ','.join(filters) + '}'
        
        # Time range - calculate exact 24h window
        now = int(time.time())
        end_s = self._floor_hour(now)
        start_s = end_s - (hours * 3600)  # Exact hours back from end
        
        # All pg_stat_statements metrics to fetch (matching Dashboard 3)
        pgss_metrics = {
            'calls': 'pgwatch_pg_stat_statements_calls',
            'exec_time_ms': 'pgwatch_pg_stat_statements_exec_time_total',
            'plan_time_ms': 'pgwatch_pg_stat_statements_plan_time_total',
            'rows': 'pgwatch_pg_stat_statements_rows',
            'shared_blks_hit_bytes': 'pgwatch_pg_stat_statements_shared_bytes_hit_total',
            'shared_blks_read_bytes': 'pgwatch_pg_stat_statements_shared_bytes_read_total',
            'shared_blks_dirtied_bytes': 'pgwatch_pg_stat_statements_shared_bytes_dirtied_total',
            'shared_blks_written_bytes': 'pgwatch_pg_stat_statements_shared_bytes_written_total',
            'wal_bytes': 'pgwatch_pg_stat_statements_wal_bytes',
            'wal_fpi': 'pgwatch_pg_stat_statements_wal_fpi',
            'wal_records': 'pgwatch_pg_stat_statements_wal_records',
            'temp_bytes_read': 'pgwatch_pg_stat_statements_temp_bytes_read',
            'temp_bytes_written': 'pgwatch_pg_stat_statements_temp_bytes_written',
            'blk_read_time_ms': 'pgwatch_pg_stat_statements_block_read_total',
            'blk_write_time_ms': 'pgwatch_pg_stat_statements_block_write_total',
            'jit_generation_time_ms': 'pgwatch_pg_stat_statements_jit_generation_time',
            'jit_inlining_time_ms': 'pgwatch_pg_stat_statements_jit_inlining_time',
            'jit_optimization_time_ms': 'pgwatch_pg_stat_statements_jit_optimization_time',
            'jit_emission_time_ms': 'pgwatch_pg_stat_statements_jit_emission_time',
        }
        
        # Fetch each metric
        for metric_key, metric_name in pgss_metrics.items():
            try:
                # Query for total increase over the time range
                query = f'sum(increase({metric_name}{filter_str}[{hours}h]))'
                result = self.query_instant(query)
                
                if result.get('status') == 'success' and result.get('data', {}).get('result'):
                    for item in result['data']['result']:
                        value = float(item['value'][1]) if item.get('value') else 0
                        # Only include metrics that have non-zero values
                        if value > 0:
                            metrics[metric_key] = value
                        break
            except Exception:
                # Silently skip metrics that fail (some may not exist for older PG versions)
                pass
        
        # Add time range info
        metrics['time_range'] = {
            'hours': hours,
            'start_time': datetime.fromtimestamp(start_s).isoformat(),
            'end_time': datetime.fromtimestamp(end_s).isoformat()
        }
        
        return metrics

    def generate_per_query_jsons(self, reports: Dict[str, Any], cluster: str,
                                 node_name: str = None,
                                 # 640 KB should be enough for anybody
                                 query_text_limit: int = 655360,
                                 hours: int = 24,
                                 write_immediately: bool = False,
                                 include_cluster_prefix: bool = True,
                                 api_url: str = None,
                                 token: str = None,
                                 report_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate individual JSON files for each query mentioned in hourly reports.
        Fetches all metrics directly from Prometheus (matching Dashboard 3).

        Args:
            reports: Dictionary of generated reports keyed by check_id
            cluster: Cluster name
            node_name: Node name (optional, will use primary if not specified)
            query_text_limit: Maximum number of characters for each query text
            hours: Number of hours for metric aggregation (default: 24)
            write_immediately: If True, write files immediately to reduce memory usage
            include_cluster_prefix: If True, prefix per-query filenames with "<cluster>_".
            api_url: API URL for uploads (only used if write_immediately is True)
            token: API token for uploads (only used if write_immediately is True)
            report_id: Report ID for uploads (only used if write_immediately is True)
            
        Returns:
            List of dictionaries with 'filename' (and optionally 'data' if not written immediately)
        """
        logger.info("Generating per-query JSON files...")
        
        # Extract all queryids from reports
        queryids_by_db = self.extract_queryids_from_reports(reports)
        
        if not queryids_by_db:
            logger.warning("No queryids found in hourly reports")
            return []

        # Determine which nodes to include (match generate_all_reports logic)
        if node_name is None:
            nodes = self.get_all_nodes(cluster)
            nodes_to_process: List[str] = []
            if nodes.get("primary"):
                nodes_to_process.append(nodes["primary"])
            nodes_to_process.extend(nodes.get("standbys", []))

            # If no nodes found, fall back to default
            if not nodes_to_process:
                logger.warning(f"No nodes found in cluster '{cluster}', using default 'node-01'")
                nodes_to_process = ["node-01"]
                nodes = {"primary": "node-01", "standbys": []}
        else:
            # Single node (backward compatibility)
            nodes_to_process = [node_name]
            nodes = {"primary": node_name, "standbys": []}
        
        # Get query texts from sink - fetch all since db names may differ between
        # prometheus (datname like 'target_database') and sink (dbname like 'target-database')
        db_names_list = list(queryids_by_db.keys())
        logger.info(f"Fetching query texts for {len(db_names_list)} database(s): {db_names_list}")
        query_texts = self.get_queryid_queries_from_sink(query_text_limit, db_names=None)
        
        query_files = []
        # Invert {db: set(queryid)} -> {queryid: set(db)}
        dbs_by_queryid: Dict[str, set] = {}
        for db_name, queryids in queryids_by_db.items():
            for qid in queryids:
                if not qid:
                    continue
                dbs_by_queryid.setdefault(qid, set()).add(db_name)

        total_queries = len(dbs_by_queryid)
        processed = 0

        # Process deterministically (helps debugging)
        for queryid in sorted(dbs_by_queryid.keys()):
            processed += 1
            dbs_for_query = sorted(list(dbs_by_queryid[queryid]))
            logger.info(f"Processing query {processed}/{total_queries}: {queryid[:20]}... (dbs={len(dbs_for_query)}, nodes={len(nodes_to_process)})")

            # Query text is expected to be identical across DBs; pick first non-empty.
            # Note: db names may differ between prometheus (datname) and sink (dbname),
            # so we search all databases in query_texts for the queryid.
            query_text = None
            for db_name in dbs_for_query:
                qt = (query_texts.get(db_name, {}) or {}).get(queryid)
                if qt:
                    query_text = qt
                    break
            # If not found by exact db match, search all dbs in sink
            if not query_text:
                for sink_db, sink_queries in query_texts.items():
                    qt = sink_queries.get(queryid)
                    if qt:
                        query_text = qt
                        break

            # Build results: results[node_name][db_name] = {"metrics": {...}}
            results_by_node: Dict[str, Dict[str, Any]] = {}
            time_range = None

            for n in nodes_to_process:
                node_block: Dict[str, Any] = {}
                for db_name in dbs_for_query:
                    metrics = self.get_query_metrics_from_prometheus(
                        cluster, n, db_name, queryid, hours=hours
                    )
                    # Pull out time_range once and keep per-db metrics clean
                    if time_range is None and isinstance(metrics, dict):
                        time_range = metrics.pop("time_range", None)
                    elif isinstance(metrics, dict):
                        metrics.pop("time_range", None)

                    node_block[db_name] = {"metrics": metrics}
                results_by_node[n] = node_block

            # Create filename (match per-check report prefix logic)
            # - Single-cluster: query_<queryid>.json
            # - Multi-cluster:  <cluster>_query_<queryid>.json
            filename = f"{cluster}_query_{queryid}.json" if include_cluster_prefix else f"query_{queryid}.json"

            # Build the final JSON object (keep timestamptz as the last field)
            now = datetime.now(timezone.utc).isoformat()
            query_data = {
                "cluster_id": cluster,
                "query_id": queryid,
                "query_text": query_text,
                "nodes": nodes,
                "results": results_by_node,
            }
            if time_range:
                query_data["time_range"] = time_range
            query_data["timestamptz"] = now

            if write_immediately:
                # Write to disk immediately to reduce memory usage
                with open(filename, "w") as f:
                    json.dump(query_data, f, indent=2)
                logger.info(f"Generated query file: {filename}")

                # Upload if API credentials provided
                if api_url and token and report_id:
                    self.upload_report_file(api_url, token, report_id, filename)

                # Only store filename, not data
                query_files.append({"filename": filename})

                # Free memory immediately after writing
                del query_data
            else:
                # Store in memory (legacy behavior)
                query_files.append({
                    "filename": filename,
                    "data": query_data
                })

            # Free memory periodically to reduce peak usage
            if processed % 10 == 0:
                gc.collect()
        
        # Final cleanup
        del query_texts
        gc.collect()
        
        logger.info(f"Generated {len(query_files)} per-query JSON files")
        return query_files

    def get_all_clusters(self) -> List[str]:
        """
        Get all unique cluster names (projects) from the metrics.
        
        Returns:
            List of cluster names
        """
        # Query for all clusters using last_over_time to get recent values
        clusters_query = 'last_over_time(pgwatch_settings_configured[3h])'
        result = self.query_instant(clusters_query)
        
        cluster_set = set()
        
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                cluster_name = item['metric'].get('cluster', '')
                if cluster_name:
                    cluster_set.add(cluster_name)
        else:
            # Debug output
            logger.info(f"Debug - get_all_clusters query status: {result.get('status')}")
            logger.info(f"Debug - get_all_clusters result count: {len(result.get('data', {}).get('result', []))}")
        
        if cluster_set:
            logger.info(f"Found {len(cluster_set)} cluster(s): {sorted(list(cluster_set))}")
        
        return sorted(list(cluster_set))

    def get_all_nodes(self, cluster: str = "local") -> Dict[str, List[str]]:
        """
        Get all nodes (primary and replicas) from the metrics.
        Uses pgwatch_db_stats_in_recovery_int to determine primary vs standby.
        
        Args:
            cluster: Cluster name
            
        Returns:
            Dictionary with 'primary' and 'standbys' keys containing node names
        """
        # Query for all nodes in the cluster using last_over_time
        nodes_query = f'last_over_time(pgwatch_settings_configured{{cluster="{cluster}"}}[3h])'
        result = self.query_instant(nodes_query)
        
        nodes = {"primary": None, "standbys": []}
        node_set = set()
        
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            for item in result['data']['result']:
                node_name = item['metric'].get('node_name', '')
                if node_name and node_name not in node_set:
                    node_set.add(node_name)
        
        # Convert to sorted list
        node_list = sorted(list(node_set))
        
        if node_list:
            logger.info(f"Found {len(node_list)} node(s) in cluster '{cluster}': {node_list}")
        else:
            logger.warning(f"No nodes found in cluster '{cluster}'")
        
        # Use pgwatch_db_stats_in_recovery_int to determine primary vs standby
        # in_recovery = 0 means primary, in_recovery = 1 means standby
        for node_name in node_list:
            recovery_query = f'last_over_time(pgwatch_db_stats_in_recovery_int{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
            recovery_result = self.query_instant(recovery_query)
            
            is_standby = False
            if recovery_result.get('status') == 'success' and recovery_result.get('data', {}).get('result'):
                if recovery_result['data']['result']:
                    in_recovery_value = float(recovery_result['data']['result'][0]['value'][1])
                    is_standby = (in_recovery_value > 0)
                    logger.info(f"Node '{node_name}': in_recovery={int(in_recovery_value)} ({'standby' if is_standby else 'primary'})")
            
            if is_standby:
                nodes["standbys"].append(node_name)
            else:
                # First non-standby node becomes primary
                if nodes["primary"] is None:
                    nodes["primary"] = node_name
                else:
                    # If we have multiple primaries (shouldn't happen), treat as replicas
                    logger.warning(f"Multiple primary nodes detected, treating '{node_name}' as replica")
                    nodes["standbys"].append(node_name)
        
        logger.info(f"Result: primary={nodes['primary']}, replicas={nodes['standbys']}")
        return nodes

    def get_all_databases(self, cluster: str = "local", node_name: str = "node-01") -> List[str]:
        """
        Get all databases from the metrics.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            
        Returns:
            List of database names
        """
        # Build a source-agnostic database list by unifying labels from:
        # 1) Generic per-database metric (wraparound) → datname
        # 2) Custom index reports (unused/redundant) → dbname
        # 3) Btree bloat (for completeness) → datname
        databases: List[str] = []
        database_set = set()

        # Helper to add a name safely
        def add_db(name: str) -> None:
            if name and name not in self.excluded_databases and name not in database_set:
                database_set.add(name)
                databases.append(name)

        # 1) Generic per-database metric
        wrap_q = f'last_over_time(pgwatch_pg_database_wraparound_age_datfrozenxid{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        wrap_res = self.query_instant(wrap_q)
        if wrap_res.get('status') == 'success' and wrap_res.get('data', {}).get('result'):
            for item in wrap_res['data']['result']:
                add_db(item["metric"].get("datname", ""))

        # 2) Custom reports - unused indexes now uses datname, redundant still uses dbname
        unused_q = f'last_over_time(pgwatch_unused_indexes_index_size_bytes{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        unused_res = self.query_instant(unused_q)
        if unused_res.get('status') == 'success' and unused_res.get('data', {}).get('result'):
            for item in unused_res['data']['result']:
                add_db(item["metric"].get("datname", ""))
        
        redun_q = f'last_over_time(pgwatch_redundant_indexes_index_size_bytes{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        redun_res = self.query_instant(redun_q)
        if redun_res.get('status') == 'success' and redun_res.get('data', {}).get('result'):
            for item in redun_res['data']['result']:
                add_db(item["metric"].get("dbname", ""))

        # 3) Btree bloat family
        bloat_q = f'last_over_time(pgwatch_pg_btree_bloat_bloat_pct{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        bloat_res = self.query_instant(bloat_q)
        if bloat_res.get('status') == 'success' and bloat_res.get('data', {}).get('result'):
            for item in bloat_res['data']['result']:
                add_db(item["metric"].get("datname", ""))
        
        # 4) pg_stat_statements metrics (calls)
        pgss_q = f'last_over_time(pgwatch_pg_stat_statements_calls{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        pgss_res = self.query_instant(pgss_q)
        if pgss_res.get('status') == 'success' and pgss_res.get('data', {}).get('result'):
            for item in pgss_res['data']['result']:
                add_db(item["metric"].get("datname", ""))
        
        # 5) Wait events
        wait_q = f'last_over_time(pgwatch_wait_events_total{{cluster="{cluster}", node_name="{node_name}"}}[3h])'
        wait_res = self.query_instant(wait_q)
        if wait_res.get('status') == 'success' and wait_res.get('data', {}).get('result'):
            for item in wait_res['data']['result']:
                add_db(item["metric"].get("datname", ""))

        return databases

    def _get_pgss_metrics_data_by_db(self, cluster: str, node_name: str, db_name: str, start_time: datetime,
                                     end_time: datetime) -> List[Dict[str, Any]]:
        """
        Get pg_stat_statements metrics data for a specific database between two time points.
        
        Args:
            cluster: Cluster name
            node_name: Node name
            db_name: Database name
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            List of query metrics with calculated differences for the specific database
        """
        # Metric name mapping for cleaner output
        METRIC_NAME_MAPPING = {
            'calls': 'calls',
            'exec_time_total': 'total_time',
            'rows': 'rows',
            'shared_bytes_hit_total': 'shared_blks_hit',
            'shared_bytes_read_total': 'shared_blks_read',
            'shared_bytes_dirtied_total': 'shared_blks_dirtied',
            'shared_bytes_written_total': 'shared_blks_written',
            'block_read_total': 'blk_read_time',
            'block_write_total': 'blk_write_time'
        }

        # Build filters including database
        filters = [f'cluster="{cluster}"', f'node_name="{node_name}"', f'datname="{db_name}"']
        filter_str = '{' + ','.join(filters) + '}'

        # Get all pg_stat_statements metrics
        all_metrics = [
            'pgwatch_pg_stat_statements_calls',
            'pgwatch_pg_stat_statements_exec_time_total',
            'pgwatch_pg_stat_statements_rows',
            'pgwatch_pg_stat_statements_shared_bytes_hit_total',
            'pgwatch_pg_stat_statements_shared_bytes_read_total',
            'pgwatch_pg_stat_statements_shared_bytes_dirtied_total',
            'pgwatch_pg_stat_statements_shared_bytes_written_total',
            'pgwatch_pg_stat_statements_block_read_total',
            'pgwatch_pg_stat_statements_block_write_total'
        ]

        # Get metrics at start and end times
        start_data = []
        end_data = []
        
        metrics_found = 0

        for metric in all_metrics:
            metric_with_filters = f'{metric}{filter_str}'

            try:
                # Query metrics around start time - use instant queries at specific timestamps
                start_result = self.query_range(metric_with_filters, start_time - timedelta(minutes=1),
                                                start_time + timedelta(minutes=1))
                if start_result:
                    start_data.extend(start_result)
                    metrics_found += 1

                # Query metrics around end time  
                end_result = self.query_range(metric_with_filters, end_time - timedelta(minutes=1),
                                              end_time + timedelta(minutes=1))
                if end_result:
                    end_data.extend(end_result)

            except Exception as e:
                logger.warning(f"Failed to query metric {metric} for database {db_name}: {e}")
                continue
        
        if metrics_found == 0:
            logger.warning(f"No pg_stat_statements metrics found for database {db_name}")
            logger.info(f"Checked time range: {start_time.isoformat()} to {end_time.isoformat()}")

        # Process the data to calculate differences
        result = self._process_pgss_data(start_data, end_data, start_time, end_time, METRIC_NAME_MAPPING)
        
        if not result:
            logger.warning(f"_process_pgss_data returned empty result for database {db_name}")
            
        return result

    def create_report(self, api_url, token, project_name, epoch):
        """
        Create a new report in the API.
        
        Args:
            api_url: API URL
            token: API token
            project_name: Project name (cluster identifier)
            epoch: Epoch identifier
            
        Returns:
            Report ID or None if creation fails
        """
        request_data = {
            "access_token": token,
            "project": project_name,
            "epoch": epoch,
        }

        try:
            response = make_request(api_url, "/rpc/checkup_report_create", request_data)
            report_id = response.get("report_id")
            if not report_id:
                message = response.get("message", "Cannot create report.")
                logger.warning(f"{message}")
                return None
            
            logger.info(f"Created report ID: {report_id}")
            return int(report_id)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if hasattr(e, 'response') else 'unknown'
            if status == 404:
                logger.warning("API endpoint not available (404). Reports will be saved locally only.")
            elif status == 400:
                logger.info(f"Request data: {len(json.dumps(request_data))} chars")
                logger.warning("API rejected request (400 Bad Request). Reports will be saved locally only.")
                logger.warning("This may indicate authentication issues or API format changes.")
            else:
                logger.error(f"Failed to create report (HTTP {status}): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            return None

    def upload_report_file(self, api_url, token, report_id, path):
        """
        Upload a report file to the API.
        
        Note: The API endpoint may not be available in all deployments.
        Use --no-upload flag to skip API uploads.
        """
        file_type = os.path.splitext(path)[1].lower().lstrip(".")
        file_name = os.path.basename(path)

        data = Path(path).read_text(encoding="utf-8")

        # Prefer extracting check_id from JSON payload (filenames vary: A002.json, cluster_A002.json, etc.)
        # Per-query JSON files intentionally do not have checkId (see reporter/schemas/query.schema.json).
        check_id = ""
        generate_issue = False
        if file_type == "json":
            try:
                payload = json.loads(data)
                if isinstance(payload, dict):
                    maybe = payload.get("checkId")
                    if isinstance(maybe, str) and maybe:
                        check_id = maybe
                        generate_issue = True
            except Exception:
                logger.warning(f"Upload: failed to parse JSON file '{file_name}', uploading without check_id")
                # Keep check_id empty / generate_issue False to avoid mislabeling.
                pass

        request_data = {
            "access_token": token,
            "checkup_report_id": report_id,
            "check_id": check_id,
            "filename": file_name,
            "data": data,
            "type": file_type,
            "generate_issue": generate_issue
        }

        try:
            # Try the primary endpoint
            response = make_request(api_url, "/rpc/checkup_report_file_post", request_data)
            if "message" in response:
                raise Exception(response["message"])
            logger.info(f"Uploaded: {file_name}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if hasattr(e, 'response') else 'unknown'
            if status == 404:
                logger.warning(f"Upload endpoint not available (404). File saved locally: {path}")
            elif status == 400:
                logger.warning(f"Upload rejected by API (400 Bad Request). File saved locally: {path}")
                logger.warning("This may indicate the API endpoint format has changed or authentication issue.")
            else:
                logger.error(f"Upload failed for {file_name} (HTTP {status}). File saved locally: {path}")
            logger.info("Use --no-upload flag to skip API uploads and suppress these warnings.")
        except Exception as e:
            logger.error(f"Upload failed for {file_name}: {e}")
            logger.info(f"File saved locally: {path}")


def make_request(api_url, endpoint, request_data):
    response = requests.post(api_url + endpoint, json=request_data)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description='Generate PostgreSQL reports using PromQL')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--prometheus-url', default='http://sink-prometheus:9090',
                        help='Prometheus URL (default: http://sink-prometheus:9090)')
    parser.add_argument('--postgres-sink-url', default='postgresql://pgwatch@sink-postgres:5432/measurements',
                        help='Postgres sink connection string (default: postgresql://pgwatch@sink-postgres:5432/measurements)')
    parser.add_argument('--cluster', default=None,
                        help='Cluster name (default: auto-detect all clusters)')
    parser.add_argument('--node-name', default=None,
                        help='Node name (default: auto-detect all nodes when combine-nodes is true)')
    parser.add_argument('--no-combine-nodes', action='store_true', default=False,
                        help='Disable combining primary and replica reports into single report')
    parser.add_argument('--check-id',
                        choices=['A002', 'A003', 'A004', 'A007', 'D004', 'F001', 'F004', 'F005', 'G001', 'H001', 'H002',
                                 'H004', 'K001', 'K003', 'K004', 'K005', 'K006', 'K007', 'K008', 'M001', 'M002', 'M003', 'N001', 'ALL'],
                        help='Specific check ID to generate (default: ALL)')
    parser.add_argument('--output', default='-',
                        help='Output file (default: stdout)')
    parser.add_argument('--api-url', default='https://postgres.ai/api/general')
    parser.add_argument('--token', default='')
    parser.add_argument('--project-name', default='project-name',
                        help='Project name for API upload (default: project-name)')
    parser.add_argument('--epoch', default='1')
    parser.add_argument('--no-upload', action='store_true', default=False,
                        help='Do not upload reports to the API')
    parser.add_argument('--exclude-databases', type=str, default=None,
                        help='Comma-separated list of additional databases to exclude from reports '
                             f'(default exclusions: {", ".join(sorted(PostgresReportGenerator.DEFAULT_EXCLUDED_DATABASES))})')
    parser.add_argument('--use-current-time', action='store_true', default=False,
                        help='Use current time instead of flooring to hour boundary. '
                             'Useful for testing with recently collected data.')

    args = parser.parse_args()
    
    # Parse excluded databases
    excluded_databases = None
    if args.exclude_databases:
        excluded_databases = [db.strip() for db in args.exclude_databases.split(',')]

    generator = PostgresReportGenerator(
        args.prometheus_url, args.postgres_sink_url, excluded_databases,
        use_current_time=args.use_current_time
    )

    # Test connection
    if not generator.test_connection():
        logger.error("Cannot connect to Prometheus. Make sure it's running and accessible.")
        sys.exit(1)

    try:
        # Discover all clusters if not specified
        clusters_to_process = []
        if args.cluster:
            clusters_to_process = [args.cluster]
        else:
            clusters_to_process = generator.get_all_clusters()
            if not clusters_to_process:
                logger.warning("No clusters found, using default 'local'")
                clusters_to_process = ['local']
            else:
                logger.info(f"Discovered clusters: {clusters_to_process}")
        
        # Process each cluster
        for cluster in clusters_to_process:
            logger.info("=" * 60)
            logger.info(f"Processing cluster: {cluster}")
            logger.info("=" * 60)
            
            # Set default node_name if not provided and not combining nodes
            combine_nodes = not args.no_combine_nodes
            if args.node_name is None and not combine_nodes:
                args.node_name = "node-01"
                
            if args.check_id == 'ALL' or args.check_id is None:
                # Generate all reports for this cluster
                report_id = None
                if not args.no_upload:
                    # Use cluster name as project name if not specified
                    project_name = args.project_name if args.project_name != 'project-name' else cluster
                    report_id = generator.create_report(args.api_url, args.token, project_name, args.epoch)
                    # If report creation failed, disable uploads for this cluster
                    if report_id is None:
                        logger.info(f"Skipping API uploads for cluster {cluster}")
                
                reports = generator.generate_all_reports(cluster, args.node_name, combine_nodes)
                
                # Generate per-query JSON files BEFORE deleting reports (needs queryids from reports)
                # Use write_immediately=True to avoid accumulating all data in memory
                logger.info("Generating per-query JSON files (streaming mode to reduce memory usage)...")
                query_files = generator.generate_per_query_jsons(
                    reports, cluster, node_name=args.node_name, 
                    # 640 KB should be enough for anybody
                    query_text_limit=66560, hours=24,
                    write_immediately=True,
                    include_cluster_prefix=(len(clusters_to_process) > 1),
                    api_url=args.api_url if (not args.no_upload and report_id) else None,
                    token=args.token if (not args.no_upload and report_id) else None,
                    report_id=report_id if (not args.no_upload and report_id) else None
                )
                
                # Clean up query files list
                del query_files
                gc.collect()
                
                # Save reports with cluster name prefix
                for report_key in list(reports.keys()):  # Use list() to avoid dict modification during iteration
                    output_filename = f"{cluster}_{report_key}.json" if len(clusters_to_process) > 1 else f"{report_key}.json"
                    with open(output_filename, "w") as f:
                        json.dump(reports[report_key], f, indent=2)
                    logger.info(f"Generated report: {output_filename}")
                    if not args.no_upload and report_id:
                        generator.upload_report_file(args.api_url, args.token, report_id, output_filename)
                    
                    # Free memory immediately after writing each report
                    del reports[report_key]
                    if len(reports) > 0 and len(reports) % 5 == 0:
                        gc.collect()
                
                # Free memory after writing all reports to disk
                del reports
                gc.collect()
            else:
                # Generate specific report - use node_name or default
                if args.node_name is None:
                    args.node_name = "node-01"

                # For D004, F001, G001 - generate A003 first and derive from it
                a003_report = None
                if args.check_id in ('D004', 'F001', 'G001'):
                    print(f"Generating A003 first for {args.check_id}...")
                    a003_report = generator.generate_a003_settings_report(cluster, args.node_name)

                if args.check_id == 'A002':
                    report = generator.generate_a002_version_report(cluster, args.node_name)
                elif args.check_id == 'A003':
                    report = generator.generate_a003_settings_report(cluster, args.node_name)
                elif args.check_id == 'A004':
                    report = generator.generate_a004_cluster_report(cluster, args.node_name)
                elif args.check_id == 'A007':
                    report = generator.generate_a007_altered_settings_report(cluster, args.node_name)
                elif args.check_id == 'D004':
                    if a003_report:
                        report = generator.generate_d004_from_a003(a003_report, cluster, args.node_name)
                    else:
                        report = generator.generate_d004_pgstat_settings_report(cluster, args.node_name)
                elif args.check_id == 'F001':
                    if a003_report:
                        report = generator.generate_f001_from_a003(a003_report, args.node_name)
                    else:
                        report = generator.generate_f001_autovacuum_settings_report(cluster, args.node_name)
                elif args.check_id == 'F004':
                    report = generator.generate_f004_heap_bloat_report(cluster, args.node_name)
                elif args.check_id == 'F005':
                    report = generator.generate_f005_btree_bloat_report(cluster, args.node_name)
                elif args.check_id == 'G001':
                    if a003_report:
                        report = generator.generate_g001_from_a003(a003_report, args.node_name)
                    else:
                        report = generator.generate_g001_memory_settings_report(cluster, args.node_name)
                elif args.check_id == 'H001':
                    report = generator.generate_h001_invalid_indexes_report(cluster, args.node_name)
                elif args.check_id == 'H002':
                    report = generator.generate_h002_unused_indexes_report(cluster, args.node_name)
                elif args.check_id == 'H004':
                    report = generator.generate_h004_redundant_indexes_report(cluster, args.node_name)
                elif args.check_id == 'K001':
                    report = generator.generate_k001_query_calls_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'K003':
                    report = generator.generate_k003_top_queries_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'K004':
                    report = generator.generate_k004_temp_bytes_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'K005':
                    report = generator.generate_k005_wal_bytes_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'K006':
                    report = generator.generate_k006_shared_read_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'K007':
                    report = generator.generate_k007_shared_hit_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'K008':
                    report = generator.generate_k008_shared_hit_read_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'M001':
                    report = generator.generate_m001_mean_time_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'M002':
                    report = generator.generate_m002_rows_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'M003':
                    report = generator.generate_m003_io_time_report(cluster, args.node_name, time_range_minutes=1440)
                elif args.check_id == 'N001':
                    report = generator.generate_n001_wait_events_report(cluster, args.node_name, hours=24)

                # Determine output filename
                base_name = f"{cluster}_{args.check_id}" if len(clusters_to_process) > 1 else args.check_id
                output_filename = f"{base_name}.json" if args.output == '-' else args.output

                # Output JSON report
                if args.output == '-' and len(clusters_to_process) == 1:
                    # Report payload to stdout must remain raw JSON (not prefixed with log metadata).
                    sys.stdout.write(json.dumps(report, indent=2) + "\n")
                else:
                    with open(output_filename, 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Report written to {output_filename}")
                    if not args.no_upload:
                        project_name = args.project_name if args.project_name != 'project-name' else cluster
                        report_id = generator.create_report(args.api_url, args.token, project_name, args.epoch)
                        if report_id:
                            generator.upload_report_file(args.api_url, args.token, report_id, output_filename)
            
            # Free memory after processing each cluster
            logger.info(f"Freeing memory after processing cluster {cluster}...")
            
            # Close and reconnect postgres to free any accumulated memory
            if generator.pg_conn:
                logger.info("Reconnecting to Postgres sink to free memory...")
                generator.close_postgres_sink()
                # Connection will be recreated on next use
            
            gc.collect()
            
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        raise e
        sys.exit(1)
    finally:
        # Clean up postgres connection
        generator.close_postgres_sink()


if __name__ == "__main__":
    main()
