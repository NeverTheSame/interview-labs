"""The settings store: env/portal precedence, candidate values, encryption at
rest, the allowlist, validation and masking. No network, no database: the
route tests' ``StubDB`` stands in for ``r3_settings``."""

import pytest

from r3 import config_registry as registry
from r3.settings_store import SecretBox, SettingsStore, SettingsUnavailable, changed_groups, mask
from tests.test_routes import StubDB

KEY_ENV = {"R3_SETTINGS_KEY": "a-long-random-settings-key"}


async def loaded(env, db=None):
    store = SettingsStore(env=dict(env))
    await store.load(db or StubDB())
    return store, db


async def test_portal_overrides_env_and_clearing_returns_to_env():
    db = StubDB()
    store, _ = await loaded({**KEY_ENV, "R3_ROUTER_MODE": "jev", "TEMPORAL_NAMESPACE": "env-ns"}, db)
    assert store.source("R3_ROUTER_MODE") == "env" and store.settings().router_mode == "jev"
    assert store.source("R3_JEV_CONFIDENCE") == "default" and store.settings().jev_confidence == 0.6

    changed = await store.save(db, {"R3_ROUTER_MODE": "rules", "TEMPORAL_NAMESPACE": "env-ns"})
    assert changed == ["R3_ROUTER_MODE"]  # saving the env value again changes nothing effective
    assert store.source("R3_ROUTER_MODE") == "portal" and store.settings().router_mode == "rules"

    changed = await store.save(db, {"R3_ROUTER_MODE": None})
    assert changed == ["R3_ROUTER_MODE"]
    assert store.source("R3_ROUTER_MODE") == "env" and store.settings().router_mode == "jev"
    assert "R3_ROUTER_MODE" not in db.settings_rows


async def test_an_empty_override_hides_an_env_value():
    db = StubDB()
    store, _ = await loaded({**KEY_ENV, "TEMPORAL_UI_URL": "https://temporal.example/ui"}, db)
    await store.save(db, {"TEMPORAL_UI_URL": ""})
    assert store.source("TEMPORAL_UI_URL") == "portal" and store.settings().temporal_ui_url == ""


async def test_candidates_layer_over_overrides_without_saving():
    db = StubDB()
    store, _ = await loaded({**KEY_ENV, "OPENAI_API_KEY": "sk-env"}, db)
    await store.save(db, {"OPENAI_API_KEY": "sk-portal"})
    assert store.settings({"OPENAI_API_KEY": "sk-typed"}).key_for("openai") == "sk-typed"
    assert store.settings({"OPENAI_API_KEY": None}).key_for("openai") == "sk-env"  # as if cleared
    assert store.settings().key_for("openai") == "sk-portal"  # nothing was saved


async def test_overrides_survive_a_restart_and_secrets_are_encrypted_at_rest():
    db = StubDB()
    store, _ = await loaded(KEY_ENV, db)
    await store.save(db, {"OPENROUTER_API_KEY": "sk-or-v1-very-secret-value", "TEMPORAL_TASK_QUEUE": "my-queue"})
    stored, secret_flag = db.settings_rows["OPENROUTER_API_KEY"]
    assert secret_flag is True and "very-secret" not in stored and stored.startswith("gAAAA")  # a Fernet token
    assert db.settings_rows["TEMPORAL_TASK_QUEUE"] == ("my-queue", False)  # non-secrets are stored in clear

    again, _ = await loaded(KEY_ENV, db)  # a new process with the same key
    assert again.overrides == {"OPENROUTER_API_KEY": "sk-or-v1-very-secret-value", "TEMPORAL_TASK_QUEUE": "my-queue"}
    assert again.settings().key_for("openrouter") == "sk-or-v1-very-secret-value"
    assert again.settings().temporal_task_queue == "my-queue"


async def test_the_database_password_is_never_an_encryption_key():
    db = StubDB()
    dsn_env = {"DATABASE_URL": "postgresql://user:db-pass-123@host/db"}
    store, _ = await loaded(dsn_env, db)
    assert store.box.source == "none" and not store.box.available
    with pytest.raises(SettingsUnavailable, match="encryption key is not configured"):
        await store.save(db, {"TEMPORAL_API_KEY": "tmprl-abcdefghijkl"})
    assert "TEMPORAL_API_KEY" not in db.settings_rows
    await store.save(db, {"TEMPORAL_TASK_QUEUE": "my-queue"})  # non-secret settings keep working
    assert db.settings_rows["TEMPORAL_TASK_QUEUE"] == ("my-queue", False)


async def test_a_secret_encrypted_with_the_old_database_password_key_is_flagged_not_decrypted():
    """Before the fallback was removed, a deploy without R3_SETTINGS_KEY saved
    secrets under a key derived from the DATABASE_URL password."""
    from r3.settings_store import _fernet_key
    from cryptography.fernet import Fernet

    db = StubDB()
    legacy = Fernet(_fernet_key("db-pass-123")).encrypt(b"tmprl-abcdefghijkl").decode()
    await db.save_setting("TEMPORAL_API_KEY", legacy, True)
    store, _ = await loaded({"DATABASE_URL": "postgresql://user:db-pass-123@host/db", **KEY_ENV}, db)
    assert "TEMPORAL_API_KEY" not in store.overrides
    assert store.undecryptable == ["TEMPORAL_API_KEY"]


async def test_a_changed_key_leaves_the_secret_undecryptable_and_falls_back_to_env():
    db = StubDB()
    store, _ = await loaded(KEY_ENV, db)
    await store.save(db, {"OPENAI_API_KEY": "sk-portal-value-0001"})
    other, _ = await loaded({"R3_SETTINGS_KEY": "a-different-key", "OPENAI_API_KEY": "sk-env-value"}, db)
    assert other.undecryptable == ["OPENAI_API_KEY"]
    assert other.source("OPENAI_API_KEY") == "env" and other.settings().key_for("openai") == "sk-env-value"
    assert "cannot be decrypted" in other.describe("OPENAI_API_KEY")["warning"]


async def test_without_key_material_secrets_are_refused_but_plain_values_save():
    db = StubDB()
    store, _ = await loaded({}, db)
    assert store.box.source == "none"
    with pytest.raises(SettingsUnavailable, match="R3_SETTINGS_KEY"):
        await store.save(db, {"OPENAI_API_KEY": "sk-x"})
    assert "OPENAI_API_KEY" not in db.settings_rows
    await store.save(db, {"R3_ROUTER_MODE": "hybrid"})
    assert store.settings().router_mode == "hybrid"


async def test_allowlist_on_validate_and_on_load():
    clean, errors = SettingsStore.validate({"DATABASE_URL": "postgres://x", "PATH": "/bin", "R3_ROUTER_MODE": "LLM"})
    assert set(errors) == {"DATABASE_URL", "PATH"} and clean == {"R3_ROUTER_MODE": "llm"}
    assert "DATABASE_URL" not in registry.BY_NAME

    db = StubDB()
    await db.save_setting("DATABASE_URL", "postgres://evil")
    await db.save_setting("R3_ROUTER_MODE", "not-a-mode")  # a stale, now-invalid value
    await db.save_setting("TEMPORAL_TASK_QUEUE", "queue")
    store, _ = await loaded(KEY_ENV, db)
    assert store.overrides == {"TEMPORAL_TASK_QUEUE": "queue"}
    assert "DATABASE_URL" not in store.merged() or store.merged()["DATABASE_URL"] != "postgres://evil"


async def test_load_never_raises():
    class Broken(StubDB):
        async def ensure_settings_table(self):
            raise PermissionError("no CREATE on schema public")

    store = SettingsStore(env=dict(KEY_ENV))
    await store.load(Broken())
    assert not store.available and "PermissionError" in store.error
    await store.load(object())  # a database without the settings mixin
    assert not store.available
    with pytest.raises(SettingsUnavailable):
        await store.save(StubDB(), {"TEMPORAL_TASK_QUEUE": "x"})


@pytest.mark.parametrize("name, raw, expected", [
    ("R3_ROUTER_MODE", " Hybrid ", "hybrid"),
    ("R3_EMBED_WORKER", "off", "0"),
    ("R3_CHAT_ALLOW_CUSTOM_MODELS", "TRUE", "1"),
    ("R3_JEV_CONFIDENCE", "0.75", "0.75"),
    ("TEMPORAL_UI_URL", "https://temporal.example/ui/", "https://temporal.example/ui"),
    ("TEMPORAL_ADDRESS", "ns.acct.tmprl.cloud:7233", "ns.acct.tmprl.cloud:7233"),
    ("R3_CHAT_MODEL", "openai:gpt-4.1-mini", "openai:gpt-4.1-mini"),
    ("R3_CHAT_MODELS", "openrouter:openai/gpt-4.1-mini=Mini, openai:gpt-4.1",
     "openrouter:openai/gpt-4.1-mini=Mini,openai:gpt-4.1"),
    ("R3_TYPESAFE_MODEL", "typesafe:jev-1", "typesafe:jev-1"),
    ("TEMPORAL_TLS_KEY", "-----BEGIN PRIVATE KEY-----\r\nabc\r\n-----END PRIVATE KEY-----\n",
     "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"),
    ("TEMPORAL_API_KEY", "tmprl abc", "tmprl abc"),
    ("TEMPORAL_TASK_QUEUE", "", ""),
])
def test_validation_normalises(name, raw, expected):
    assert registry.validate(name, raw) == expected


@pytest.mark.parametrize("name, raw", [
    ("R3_ROUTER_MODE", "smart"),
    ("R3_EMBED_WORKER", "maybe"),
    ("R3_JEV_CONFIDENCE", "1.5"),
    ("TEMPORAL_UI_URL", "ftp://temporal.example"),
    ("TEMPORAL_ADDRESS", "no-port"),
    ("R3_CHAT_MODEL", "gpt-4"),
    ("R3_CHAT_MODEL", "typesafe:jev-latest"),
    ("R3_TYPESAFE_MODEL", "openai:gpt-4.1"),
    ("R3_CHAT_MODELS", "openai:gpt-4.1,bad"),
    ("OPENAI_API_KEY", "sk-one\nsk-two"),
    ("TEMPORAL_TLS_CERT", "not pem\nbut two lines"),
    ("TEMPORAL_TASK_QUEUE", "a" * 1001),
])
def test_validation_rejects(name, raw):
    with pytest.raises(ValueError):
        registry.validate(name, raw)


def test_masking_never_shows_more_than_four_characters():
    assert mask("sk-or-v1-0123456789abcdef") == "••••cdef"
    assert mask("short-key") == "••••"  # too short to reveal any of it
    assert mask("-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----") == "PEM key"
    assert mask("/etc/r3/client.key", pem=True) == "file path"


def test_describe_never_carries_a_secret():
    store = SettingsStore(env={"OPENAI_API_KEY": "sk-proj-abcdefghijklmnop", "TEMPORAL_TASK_QUEUE": "queue"})
    d = store.describe("OPENAI_API_KEY")
    assert d["value"] is None and d["masked"] == "••••mnop" and d["set"] and d["source"] == "env"
    assert "abcdefghijkl" not in repr(store.state())
    assert store.describe("TEMPORAL_TASK_QUEUE")["value"] == "queue"


def test_registry_shape():
    names = [s.name for s in registry.SETTINGS]
    assert len(names) == len(set(names))
    assert {s.group for s in registry.SETTINGS} <= set(registry.GROUPS_BY_ID)
    assert {s.test for s in registry.SETTINGS if s.test} <= set(registry.TESTS_BY_ID)
    assert registry.SECRET_NAMES == {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY", "TEMPORAL_API_KEY",
                                     "TEMPORAL_TLS_KEY"}
    assert changed_groups(["OPENAI_API_KEY", "TEMPORAL_ADDRESS", "nope"]) == {"models", "temporal"}


def test_secret_box_round_trip():
    box = SecretBox("k" * 10)
    token = box.encrypt("hello")
    assert token != "hello" and box.decrypt(token) == "hello"
    assert SecretBox("   ").source == "none" and not SecretBox("   ").available
