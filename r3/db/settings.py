"""Queries over ``r3_settings``: the values an operator entered on the
Settings page. R3 owns this table, so it is created on demand; secrets
arrive here already encrypted by :mod:`r3.settings_store`, which also
enforces the allowlist."""

from __future__ import annotations

from ._base import QueryBase

SETTINGS_DDL = """
CREATE TABLE IF NOT EXISTS r3_settings (
    key        text PRIMARY KEY,
    value      text NOT NULL,
    secret     boolean NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
)"""


class SettingsQueries(QueryBase):
    async def ensure_settings_table(self) -> None:
        """Create ``r3_settings`` when it does not exist yet."""
        await self.pool.execute(SETTINGS_DDL)

    async def load_settings(self) -> dict[str, str]:
        """Every stored override, name -> stored value (ciphertext for secrets)."""
        rows = await self.pool.fetch("SELECT key, value FROM r3_settings")
        return {r["key"]: r["value"] for r in rows}

    async def save_setting(self, key: str, value: str, secret: bool = False) -> None:
        await self.pool.execute(
            """INSERT INTO r3_settings (key, value, secret, updated_at) VALUES ($1, $2, $3, now())
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, secret = EXCLUDED.secret,
                                               updated_at = now()""",
            key, value, secret)

    async def delete_setting(self, key: str) -> None:
        await self.pool.execute("DELETE FROM r3_settings WHERE key = $1", key)
