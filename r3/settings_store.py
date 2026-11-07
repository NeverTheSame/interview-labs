"""``SettingsStore``: the environment plus the overrides an operator saved on
the Settings page, merged into one mapping that
:meth:`ControllerSettings.from_env` reads.

Precedence is simple: a portal override wins over the environment, and
clearing the override returns the name to its environment value (or the
built-in default). Only names in :mod:`r3.config_registry` can be stored.

Secrets are stored encrypted with Fernet (AES-128-CBC + HMAC-SHA256, from
``cryptography``). The key is SHA-256 of ``R3_SETTINGS_KEY``, urlsafe-base64
encoded as Fernet expects. There is no fallback: without ``R3_SETTINGS_KEY``
no secret can be saved (non-secret settings still can) and stored secrets are
reported as undecryptable. In particular the ``DATABASE_URL`` password, which
anyone with database access already has, is never used as a key. Secrets saved
by an earlier version under that derived key are not readable now; the page
flags them and they are entered again once the key is set.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Protocol

from cryptography.fernet import Fernet, InvalidToken

from r3 import config_registry as registry
from r3.settings import ControllerSettings

log = logging.getLogger("r3.settings_store")

Source = Literal["portal", "env", "default"]
KEY_CONTEXT = b"r3-settings/v1:"


class SettingsDB(Protocol):
    async def ensure_settings_table(self) -> None: ...

    async def load_settings(self) -> dict[str, str]: ...

    async def save_setting(self, key: str, value: str, secret: bool = False) -> None: ...

    async def delete_setting(self, key: str) -> None: ...


class SettingsUnavailable(Exception):
    """Overrides cannot be stored (no settings table, or settings pinned)."""


def _fernet_key(material: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(KEY_CONTEXT + material.encode()).digest())


class SecretBox:
    """Encrypts and decrypts stored secrets. ``source`` is ``R3_SETTINGS_KEY``
    when that key is configured and ``none`` when it is not (secrets cannot be
    stored)."""

    def __init__(self, settings_key: str = "") -> None:
        self.source = "none"
        self._box: Fernet | None = None
        if settings_key.strip():
            self._box = Fernet(_fernet_key(settings_key.strip()))
            self.source = "R3_SETTINGS_KEY"

    @property
    def available(self) -> bool:
        return self._box is not None

    def encrypt(self, value: str) -> str:
        if self._box is None:
            raise SettingsUnavailable("Secrets cannot be stored: the encryption key is not configured. "
                                      "Set R3_SETTINGS_KEY to a long random value and restart the portal.")
        return self._box.encrypt(value.encode()).decode()

    def decrypt(self, token: str) -> str:
        if self._box is None:
            raise InvalidToken
        return self._box.decrypt(token.encode()).decode()


@dataclass
class SettingsStore:
    """Holds the environment snapshot and the portal overrides.

    ``env`` is the process environment at startup (``.env`` included);
    ``overrides`` are the decrypted portal values. ``available`` is whether
    overrides can be saved; ``error`` says why not."""

    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    dsn: str = ""
    overrides: dict[str, str] = field(default_factory=dict)
    available: bool = False
    error: str = "Portal overrides are not loaded yet."
    undecryptable: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.box = SecretBox(self.env.get("R3_SETTINGS_KEY", ""))

    # --- reading -------------------------------------------------------------

    def merged(self, candidates: Mapping[str, str | None] | None = None) -> dict[str, str]:
        """The environment with the overrides on top, then ``candidates``
        (unsaved form values: a string replaces, ``None`` drops the override
        so the environment value shows through)."""
        out = dict(self.env)
        out.update(self.overrides)
        for name, value in (candidates or {}).items():
            if value is None:
                if name in self.env:
                    out[name] = self.env[name]
                else:
                    out.pop(name, None)
            else:
                out[name] = value
        return out

    def settings(self, candidates: Mapping[str, str | None] | None = None) -> ControllerSettings:
        return ControllerSettings.from_env(self.merged(candidates))

    def source(self, name: str) -> Source:
        if name in self.overrides:
            return "portal"
        if self.env.get(name, "").strip():
            return "env"
        return "default"

    def value(self, name: str) -> str:
        if name in self.overrides:
            return self.overrides[name]
        return self.env.get(name, "").strip()

    # --- loading and saving ------------------------------------------------------

    async def load(self, database: Any) -> None:
        """Read the overrides from ``database``. Never raises: when the table
        cannot be created or read, the portal runs on the environment and
        ``error`` says why."""
        if not all(hasattr(database, m) for m in ("ensure_settings_table", "load_settings", "save_setting",
                                                   "delete_setting")):
            self.available, self.error = False, "This database has no settings table support."
            return
        try:
            await database.ensure_settings_table()
            rows = await database.load_settings()
        except Exception as e:
            log.warning("portal settings unavailable: %s: %s", type(e).__name__, e)
            self.available, self.error = False, f"Could not read r3_settings: {type(e).__name__}"
            return
        self.overrides, self.undecryptable = {}, []
        for name, stored in rows.items():
            s = registry.BY_NAME.get(name)
            if s is None:
                log.warning("ignoring stored setting %s: not in the registry", name)
                continue
            if s.secret:
                try:
                    stored = self.box.decrypt(stored)
                except (InvalidToken, ValueError):
                    log.warning("stored secret %s cannot be decrypted (key changed?); using the environment", name)
                    self.undecryptable.append(name)
                    continue
            try:
                self.overrides[name] = registry.validate(name, stored)
            except ValueError as e:
                log.warning("ignoring stored setting %s: %s", name, e)
        self.available, self.error = True, ""

    @staticmethod
    def validate(values: Mapping[str, object]) -> tuple[dict[str, str | None], dict[str, str]]:
        """Normalised values and per-name errors. Unknown names are errors."""
        clean: dict[str, str | None] = {}
        errors: dict[str, str] = {}
        for name, raw in values.items():
            if name not in registry.BY_NAME:
                errors[name] = "is not a setting the portal manages"
                continue
            if raw is None:
                clean[name] = None
                continue
            try:
                clean[name] = registry.validate(name, raw)
            except ValueError as e:
                errors[name] = str(e)
        return clean, errors

    async def save(self, database: SettingsDB, values: Mapping[str, str | None]) -> list[str]:
        """Store (or, for ``None``, delete) already-validated values and
        return the names whose effective value changed."""
        if not self.available:
            raise SettingsUnavailable(self.error or "Portal overrides are unavailable.")
        if any(v is not None and registry.BY_NAME[k].secret for k, v in values.items()) and not self.box.available:
            self.box.encrypt("")  # raises SettingsUnavailable with the reason
        before = self.merged()
        for name, value in values.items():
            s = registry.BY_NAME[name]
            if value is None:
                await database.delete_setting(name)
                self.overrides.pop(name, None)
            else:
                await database.save_setting(name, self.box.encrypt(value) if s.secret else value, s.secret)
                self.overrides[name] = value
            if name in self.undecryptable:
                self.undecryptable.remove(name)
        after = self.merged()
        return [n for n in values if before.get(n) != after.get(n)]

    # --- what the page shows ---------------------------------------------------------

    def describe(self, name: str) -> dict[str, Any]:
        s = registry.BY_NAME[name]
        value = self.value(name)
        out: dict[str, Any] = {
            "name": s.name, "label": s.label, "help": s.help, "kind": s.kind, "secret": s.secret,
            "set": bool(value), "source": self.source(name), "default": s.default, "test": s.test or None,
            "options": list(s.options) if s.options else None,
        }
        if s.secret:
            out["masked"] = mask(value, pem=s.kind == "pem") if value else ""
            out["value"] = None
        else:
            out["masked"] = None
            out["value"] = value
        if name in self.undecryptable:
            out["warning"] = "The value saved in the portal cannot be decrypted (the key changed); save it again."
        return out

    def state(self) -> dict[str, Any]:
        groups = []
        for g in registry.GROUPS:
            groups.append({
                "id": g.id, "title": g.title, "description": g.description,
                "tests": [{"id": t.id, "label": t.label, "describes": t.describes,
                           "configured": self.test_configured(t.id)}
                          for t in registry.tests_for(g.id)],
                "group_tests": list(g.tests),
                "keys": [self.describe(s.name) for s in registry.settings_in(g.id)],
            })
        return {"groups": groups,
                "store": {"available": self.available, "error": self.error, "encryption": self.box.source}}

    def test_configured(self, test_id: str, candidates: Mapping[str, str | None] | None = None) -> bool:
        merged = self.merged(candidates)
        return all(merged.get(n, "").strip() for n in registry.TESTS_BY_ID[test_id].needs)

    def secret_values(self, candidates: Mapping[str, str | None] | None = None) -> list[str]:
        """Every secret value in play, so messages can be scrubbed of them."""
        merged = self.merged(candidates)
        return [v for n in registry.SECRET_NAMES if len(v := merged.get(n, "").strip()) >= 4]


def mask(value: str, *, pem: bool = False) -> str:
    """What the page shows for a secret: never more than its last four
    characters, and none of a short one."""
    value = value.strip()
    if pem or value.startswith("-----BEGIN"):
        return "PEM key" if value.startswith("-----BEGIN") else "file path"
    if len(value) < 12:
        return "••••"
    return f"••••{value[-4:]}"


def changed_groups(names: Iterable[str]) -> set[str]:
    return {registry.BY_NAME[n].group for n in names if n in registry.BY_NAME}
