"""Configuration schema for Ziggurat."""

from __future__ import annotations

import voluptuous as vol
from zigpy.config import CONFIG_SCHEMA as ZIGPY_CONFIG_SCHEMA

CONF_ZIGGURAT_CONFIG = "ziggurat_config"
CONF_TUNABLES = "tunables"

CONFIG_SCHEMA = ZIGPY_CONFIG_SCHEMA.extend(
    {
        vol.Optional(CONF_ZIGGURAT_CONFIG, default={}): vol.Schema(
            {
                vol.Optional(CONF_TUNABLES, default={}): vol.Schema(
                    {vol.Optional(str): int}
                )
            }
        ),
    }
)
