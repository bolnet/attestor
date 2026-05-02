"""Tests for layered cloud connection config."""

import os
from unittest.mock import patch

from attestor.store.connection import (
    AuthConfig,
    CloudConnection,
    TLSConfig,
    build_config,
    merge_config_layers,
    parse_url,
    resolve_env,
    resolve_env_recursive,
    _normalize_flat_auth,
)


class TestResolveEnv:
    def test_plain_string(self):
        assert resolve_env("hello") == "hello"

    def test_dollar_var(self):
        with patch.dict(os.environ, {"MY_VAR": "resolved"}):
            assert resolve_env("$MY_VAR") == "resolved"

    def test_braced_var(self):
        with patch.dict(os.environ, {"MY_VAR": "resolved"}):
            assert resolve_env("${MY_VAR}") == "resolved"

    def test_unset_var_returns_original(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_env("$UNSET_VAR") == "$UNSET_VAR"

    def test_non_string(self):
        assert resolve_env(42) == 42
        assert resolve_env(None) is None
        assert resolve_env(True) is True

    def test_recursive(self):
        with patch.dict(os.environ, {"DB_URL": "https://cloud.example.com"}):
            data = {
                "url": "$DB_URL",
                "port": 8529,
                "auth": {"password": "$DB_URL"},
                "tags": ["$DB_URL", "literal"],
            }
            resolved = resolve_env_recursive(data)
            assert resolved["url"] == "https://cloud.example.com"
            assert resolved["port"] == 8529
            assert resolved["auth"]["password"] == "https://cloud.example.com"
            assert resolved["tags"] == ["https://cloud.example.com", "literal"]


class TestMergeConfigLayers:
    def test_single_layer(self):
        assert merge_config_layers({"a": 1}) == {"a": 1}

    def test_override(self):
        result = merge_config_layers({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_deep_merge(self):
        base = {"auth": {"username": "root", "password": ""}}
        override = {"auth": {"password": "secret"}}
        result = merge_config_layers(base, override)
        assert result == {"auth": {"username": "root", "password": "secret"}}

    def test_none_layers_skipped(self):
        result = merge_config_layers({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_layers_skipped(self):
        result = merge_config_layers({"a": 1}, {}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_three_layers(self):
        result = merge_config_layers(
            {"mode": "cloud", "port": 8529},
            {"port": 9999, "database": "test"},
            {"database": "prod"},
        )
        assert result == {"mode": "cloud", "port": 9999, "database": "prod"}


class TestNormalizeFlatAuth:
    def test_flat_to_nested(self):
        config = {"username": "root", "password": "pw", "mode": "local"}
        result = _normalize_flat_auth(config)
        assert result == {
            "mode": "local",
            "auth": {"username": "root", "password": "pw"},
        }

    def test_nested_takes_priority(self):
        config = {
            "username": "flat_user",
            "auth": {"username": "nested_user", "password": "nested_pw"},
        }
        result = _normalize_flat_auth(config)
        assert result["auth"]["username"] == "nested_user"
        assert result["auth"]["password"] == "nested_pw"

    def test_no_flat_fields(self):
        config = {"mode": "cloud", "url": "https://..."}
        result = _normalize_flat_auth(config)
        assert result == config


class TestParseUrl:
    def test_postgresql_url(self):
        result = parse_url("postgresql://user:pass@rds.aws.com:5432/mydb")
        assert result["_engine"] == "postgres"
        assert result["database"] == "mydb"
        assert result["auth"]["username"] == "user"

    def test_neo4j_secure(self):
        result = parse_url("neo4j+s://user:pw@aura.neo4j.io/mydb")
        assert result["_engine"] == "neo4j"
        assert result["tls"]["verify"] is True
        assert result["url"] == "https://aura.neo4j.io"

    def test_query_params(self):
        result = parse_url("postgresql://u:p@h:5432/db?sslmode=require&timeout=30")
        assert result["options"]["sslmode"] == "require"
        assert result["options"]["timeout"] == "30"

    def test_url_encoded_password(self):
        result = parse_url("postgresql://postgres:p%40ss%23word@host:5432/db")
        assert result["auth"]["password"] == "p@ss#word"

    def test_no_port(self):
        result = parse_url("postgresql://user:pass@host/db")
        assert result["url"] == "http://host"
        assert "port" not in result

    def test_bolt_scheme(self):
        result = parse_url("bolt://neo4j:pass@localhost:7687/mydb")
        assert result["_engine"] == "neo4j"


class TestAuthConfig:
    def test_from_dict(self):
        with patch.dict(os.environ, {"PW": "secret"}):
            auth = AuthConfig.from_dict({"username": "admin", "password": "$PW"})
            assert auth.username == "admin"
            assert auth.password == "secret"

    def test_has_credentials(self):
        assert AuthConfig(username="u").has_credentials
        assert AuthConfig(token="t").has_credentials
        assert AuthConfig(api_key="k").has_credentials
        assert not AuthConfig().has_credentials


class TestTLSConfig:
    def test_defaults(self):
        tls = TLSConfig()
        assert tls.verify is True
        assert tls.ca_cert is None

    def test_from_dict(self):
        tls = TLSConfig.from_dict({"verify": False, "ca_cert": "/path/ca.pem"})
        assert tls.verify is False
        assert tls.ca_cert == "/path/ca.pem"


class TestBuildConfig:
    def test_engine_defaults_applied(self):
        result = build_config("postgres", {})
        assert result["port"] == 5432
        assert result["auth"]["username"] == "postgres"

    def test_user_config_overrides_defaults(self):
        result = build_config("postgres", {"database": "custom"})
        assert result["database"] == "custom"
        assert result["port"] == 5432  # default preserved

    def test_cli_overrides_user_config(self):
        result = build_config(
            "postgres",
            {"database": "user_db"},
            cli_overrides={"database": "cli_db"},
        )
        assert result["database"] == "cli_db"

    def test_url_string_parsed(self):
        result = build_config(
            "postgres",
            {"url": "postgresql://admin:secret@cloud.host:5432/mydb"},
        )
        assert result["database"] == "mydb"
        assert result["auth"]["username"] == "admin"
        assert result["auth"]["password"] == "secret"

    def test_explicit_fields_override_url(self):
        result = build_config(
            "postgres",
            {
                "url": "postgresql://postgres:urlpw@host:5432/urldb",
                "database": "explicit_db",
            },
        )
        # Explicit database field overrides URL-parsed database
        assert result["database"] == "explicit_db"

    def test_env_var_in_url(self):
        with patch.dict(os.environ, {"DB_HOST": "cloud.host.com"}):
            result = build_config(
                "postgres",
                {"url": "postgresql://$DB_HOST:5432", "mode": "cloud"},
            )
            # env not resolved yet at this stage (CloudConnection does it)
            assert "DB_HOST" in str(result["url"]) or "cloud.host.com" in str(result["url"])

    def test_neo4j_defaults(self):
        result = build_config("neo4j", {})
        assert result["port"] == 7687
        assert result["auth"]["username"] == "neo4j"

    def test_unknown_engine_no_crash(self):
        result = build_config("unknown_engine", {"url": "http://localhost"})
        assert result["url"] == "http://localhost"


class TestCloudConnection:
    def test_defaults(self):
        conn = CloudConnection.from_config({})
        assert conn.mode == "cloud"
        assert conn.database == "attestor"
        assert conn.tls.verify is True

    def test_env_resolved(self):
        with patch.dict(os.environ, {"MY_URL": "https://cloud.example.com"}):
            conn = CloudConnection.from_config({"url": "$MY_URL"})
            assert conn.url == "https://cloud.example.com"

    def test_local_mode_overrides_url(self):
        conn = CloudConnection.from_config({"mode": "local", "port": 9999})
        assert conn.url == "http://localhost:9999"

    def test_nested_auth(self):
        with patch.dict(os.environ, {"PW": "s3cret"}):
            conn = CloudConnection.from_config({
                "auth": {"username": "admin", "password": "$PW"}
            })
            assert conn.auth.username == "admin"
            assert conn.auth.password == "s3cret"

    def test_flat_auth_backward_compat(self):
        conn = CloudConnection.from_config({
            "username": "oldstyle",
            "password": "oldpw",
        })
        assert conn.auth.username == "oldstyle"
        assert conn.auth.password == "oldpw"

    def test_engine_defaults_applied(self):
        conn = CloudConnection.from_config({}, backend_name="postgres")
        assert conn.auth.username == "postgres"
        assert conn.port == 5432

    def test_project_config_overrides_engine_defaults(self):
        conn = CloudConnection.from_config(
            {"auth": {"username": "myuser"}},
            backend_name="postgres",
        )
        assert conn.auth.username == "myuser"

    def test_extra_fields_preserved(self):
        conn = CloudConnection.from_config({
            "custom_option": True,
            "region": "us-east-1",
        })
        assert conn.extra["custom_option"] is True
        assert conn.extra["region"] == "us-east-1"

    def test_url_string_to_connection(self):
        conn = CloudConnection.from_config(
            {"url": "postgresql://admin:pw@host:5432/mydb"},
            backend_name="postgres",
        )
        assert conn.database == "mydb"
        assert conn.auth.username == "admin"
        assert conn.auth.password == "pw"

    def test_full_layered_example(self):
        """Simulates: engine defaults → project config → env → CLI."""
        with patch.dict(os.environ, {"DB_PW": "prod_secret"}):
            conn = CloudConnection.from_config(
                {
                    "mode": "cloud",
                    "database": "myproject",
                    "url": "postgresql://db.example.com:5432",
                    "auth": {"password": "$DB_PW"},
                },
                backend_name="postgres",
            )

        assert conn.database == "myproject"
        assert conn.auth.username == "postgres"     # from engine defaults
        assert conn.auth.password == "prod_secret"  # from env
        assert conn.url == "postgresql://db.example.com:5432"

    def test_postgres_connection(self):
        conn = CloudConnection.from_config(
            {"url": "postgresql://myuser:mypass@rds.aws.com:5432/proddb"},
            backend_name="postgres",
        )
        assert conn.database == "proddb"
        assert conn.auth.username == "myuser"
        assert conn.port == 5432
