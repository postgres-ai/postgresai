import json
import time
from unittest.mock import patch
import app
import pytest

@pytest.fixture(autouse=True)
def no_unrelated_startup_migration():
    with patch('app._apply_trigger_migration'):
        yield

def scope():
    return dict(cluster='c', node_name='n', datname='d', queryid='9223372036854775807', start=time.time()-120, end=time.time()-1)

def test_native_query_text_preserves_original_and_parameterizes_all_scope():
    body = scope()
    text = '/* header */\nSELECT  $1, \'☃\'\nFROM items'
    with patch('app.psycopg2.connect') as connect:
        cursor = connect.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = dict(query=text, observed_at=body['end'])
        response = app.app.test_client().post('/query_text', json=body)
        assert response.status_code == 200
        assert response.json['query'] == text
        assert response.json['truncated'] is False
        sql, params = cursor.execute.call_args.args
        assert params == tuple(body[k] for k in ('cluster','node_name','datname','queryid','start','end'))
        assert 'having count(distinct query) = 1' in sql

def test_native_query_text_caps_utf8_and_reports_missing():
    with patch('app.psycopg2.connect') as connect:
        cursor = connect.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = dict(query='☃'*30000, observed_at=scope()['end'])
        response = app.app.test_client().post('/query_text', json=scope())
        assert response.status_code == 200
        assert len(response.json['query'].encode()) <= 65536
        assert response.json['truncated'] is True
        cursor.fetchone.return_value = None
        assert app.app.test_client().post('/query_text', json=scope()).json['available'] is False

def test_native_query_text_rejects_unbounded_foreign_fields_without_connecting():
    with patch('app.psycopg2.connect') as connect:
        for change in [dict(sql='select secret'),dict(queryid="1' or true"),dict(cluster=''),dict(start=0),dict(end=time.time()+100),dict(start=True)]:
            body=scope(); body.update(change)
            assert app.app.test_client().post('/query_text', json=body).status_code == 400
        connect.assert_not_called()


def test_latest_collected_is_explicit_and_uses_database_lookup_clock():
    body = scope(); body['lookup_mode'] = 'latest_collected'
    with patch('app.psycopg2.connect') as connect:
        cursor = connect.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = dict(query='SELECT  $1', observed_at=body['end']+0.5, retrieved_at=body['end']+1)
        response = app.app.test_client().post('/query_text', json=body)
        assert response.status_code == 200
        assert response.json['text_scope'] == 'latest_collected'
        assert response.json['observed_at'] > body['end']
        assert response.json['retrieved_at'] >= response.json['observed_at']
        sql, params = cursor.execute.call_args.args
        assert 'statement_timestamp()' in sql
        assert "interval '93 days'" in sql
        assert 'time <= bounds.retrieved_at' in sql
        assert params == tuple(body[k] for k in ('cluster','node_name','datname','queryid','start'))
    body['lookup_mode'] = 'arbitrary'
    assert app.app.test_client().post('/query_text', json=body).status_code == 400
