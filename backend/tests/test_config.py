"""Configuration behaviour.

Two classes of bug motivated these tests:

* ``Field(env="...")`` is silently ignored by pydantic-settings v2, so
  documented variable names such as ``JWT_ALGORITHM`` had no effect.
* Nothing stopped the service booting in production with the example secret
  key and authentication switched off.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.core.config import Settings


def build(**overrides) -> Settings:
    """Construct settings from explicit values, ignoring any .env file."""
    defaults = {
        "environment": "development",
        "secret_key": "x" * 48,
        "_env_file": None,
    }
    return Settings(**{**defaults, **overrides})


class TestDocumentedVariablesTakeEffect:
    """Every variable named in .env.example must actually be read."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("embedding_model", "all-mpnet-base-v2", "all-mpnet-base-v2"),
            ("vector_store_path", "/data/vectors", "/data/vectors"),
            ("chunk_size", 512, 512),
            ("chunk_overlap", 64, 64),
            ("max_conversation_history", 25, 25),
        ],
    )
    def test_rag_settings_are_configurable(self, field, value, expected):
        assert getattr(build(**{field: value}), field) == expected

    def test_jwt_algorithm_env_var_is_honoured(self, monkeypatch):
        """Previously declared as Field(env="JWT_ALGORITHM") and never read."""
        monkeypatch.setenv("JWT_ALGORITHM", "HS512")
        settings = Settings(secret_key="x" * 48, _env_file=None)
        assert settings.algorithm == "HS512"

    def test_app_version_env_var_is_honoured(self, monkeypatch):
        monkeypatch.setenv("APP_VERSION", "9.9.9")
        settings = Settings(secret_key="x" * 48, _env_file=None)
        assert settings.version == "9.9.9"


class TestListParsing:
    def test_comma_separated_origins_become_a_list(self):
        settings = build(allowed_origins="http://a.test, http://b.test")
        assert settings.allowed_origins == ["http://a.test", "http://b.test"]

    def test_json_array_origins_are_accepted(self):
        settings = build(allowed_origins='["http://a.test"]')
        assert settings.allowed_origins == ["http://a.test"]

    def test_blank_entries_are_dropped(self):
        assert build(allowed_hosts="a.test,,b.test").allowed_hosts == [
            "a.test",
            "b.test",
        ]


class TestValidation:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        """Otherwise the splitter never advances and chunking loops."""
        with pytest.raises(ValidationError, match="chunk_overlap"):
            build(chunk_size=500, chunk_overlap=500)

    def test_unknown_log_level_is_rejected(self):
        with pytest.raises(ValidationError):
            build(log_level="CHATTY")

    def test_log_level_is_normalised_to_upper_case(self):
        assert build(log_level="debug").log_level == "DEBUG"

    def test_unknown_environment_is_rejected(self):
        with pytest.raises(ValidationError):
            build(environment="prd")


class TestProductionHardening:
    """Unsafe production configurations must fail at startup, not silently."""

    def test_placeholder_secret_key_is_refused(self):
        with pytest.raises(ValidationError, match="placeholder"):
            Settings(
                environment="production",
                debug=False,
                secret_key="your-secret-key-change-in-production",
                allowed_hosts=["api.example.com"],
                allowed_origins=["https://app.example.com"],
                _env_file=None,
            )

    def test_short_secret_key_is_refused(self):
        with pytest.raises(ValidationError, match="at least 32 characters"):
            Settings(
                environment="production",
                debug=False,
                secret_key="tooshort",
                allowed_hosts=["api.example.com"],
                allowed_origins=["https://app.example.com"],
                _env_file=None,
            )

    def test_disabling_authentication_is_refused(self):
        with pytest.raises(ValidationError, match="REQUIRE_AUTHENTICATION"):
            Settings(
                environment="production",
                debug=False,
                secret_key="x" * 48,
                require_authentication=False,
                allowed_hosts=["api.example.com"],
                allowed_origins=["https://app.example.com"],
                _env_file=None,
            )

    def test_wildcard_hosts_are_refused(self):
        with pytest.raises(ValidationError, match="ALLOWED_HOSTS"):
            Settings(
                environment="production",
                debug=False,
                secret_key="x" * 48,
                allowed_hosts=["*"],
                allowed_origins=["https://app.example.com"],
                _env_file=None,
            )

    def test_wildcard_origins_are_refused(self):
        """'*' with allow_credentials=True is the classic CORS mistake."""
        with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
            Settings(
                environment="production",
                debug=False,
                secret_key="x" * 48,
                allowed_hosts=["api.example.com"],
                allowed_origins=["*"],
                _env_file=None,
            )

    def test_debug_mode_is_refused(self):
        with pytest.raises(ValidationError, match="DEBUG"):
            Settings(
                environment="production",
                debug=True,
                secret_key="x" * 48,
                allowed_hosts=["api.example.com"],
                allowed_origins=["https://app.example.com"],
                _env_file=None,
            )

    def test_a_correct_production_config_starts(self):
        settings = Settings(
            environment="production",
            debug=False,
            secret_key="x" * 48,
            allowed_hosts=["api.example.com"],
            allowed_origins=["https://app.example.com"],
            _env_file=None,
        )
        assert settings.is_production
        assert settings.docs_enabled is False

    def test_development_tolerates_the_example_secret(self):
        """Local runs must not require ceremony."""
        settings = build(secret_key="your-secret-key-change-in-production")
        assert settings.environment == "development"
        assert settings.docs_enabled is True
