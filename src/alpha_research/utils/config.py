"""
Configuration management for Alpha Research Trading System.

Provides centralized access to all configuration files with validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, validator
from datetime import time


# Global config cache
_config_cache: Dict[str, Any] = {}


class ConfigPaths:
    """Standard paths for configuration files."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to project root (src/alpha_research/utils/config.py -> project root)
            # Go up: utils -> alpha_research -> src -> project_root
            base_dir = Path(__file__).parent.parent.parent.parent
        self.base_dir = base_dir
        self.config_dir = base_dir / "config"

    @property
    def settings(self) -> Path:
        return self.config_dir / "settings.yaml"

    @property
    def risk_limits(self) -> Path:
        return self.config_dir / "risk_limits.yaml"

    @property
    def governance_policy(self) -> Path:
        return self.config_dir / "governance_policy.yaml"

    @property
    def factor_defs(self) -> Path:
        return self.config_dir / "factor_defs.yaml"

    @property
    def universe_rules(self) -> Path:
        return self.config_dir / "universe_rules.yaml"

    @property
    def execution_policy(self) -> Path:
        return self.config_dir / "execution_policy.yaml"

    @property
    def monitoring(self) -> Path:
        return self.config_dir / "monitoring.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(config_name: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a configuration file by name.

    Args:
        config_name: One of 'settings', 'risk_limits', 'governance_policy',
                    'factor_defs', 'universe_rules', 'execution_policy', 'monitoring'
        base_dir: Optional base directory for config files

    Returns:
        Dictionary containing the configuration
    """
    paths = ConfigPaths(base_dir)

    config_map = {
        'settings': paths.settings,
        'risk_limits': paths.risk_limits,
        'governance_policy': paths.governance_policy,
        'factor_defs': paths.factor_defs,
        'universe_rules': paths.universe_rules,
        'execution_policy': paths.execution_policy,
        'monitoring': paths.monitoring,
    }

    if config_name not in config_map:
        raise ValueError(f"Unknown config name: {config_name}. "
                        f"Valid options: {list(config_map.keys())}")

    cache_key = f"{base_dir}:{config_name}"
    if cache_key not in _config_cache:
        _config_cache[cache_key] = load_yaml(config_map[config_name])

    return _config_cache[cache_key]


def get_config(config_name: str, *keys: str, default: Any = None) -> Any:
    """
    Get a specific value from a configuration file.

    Args:
        config_name: Name of the configuration file
        *keys: Path of keys to navigate to the value
        default: Default value if key not found

    Returns:
        The configuration value or default

    Example:
        >>> get_config('settings', 'portfolio', 'target_holdings')
        25
    """
    config = load_config(config_name)

    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def reload_config(config_name: Optional[str] = None) -> None:
    """
    Clear configuration cache to force reload.

    Args:
        config_name: Specific config to reload, or None for all
    """
    global _config_cache

    if config_name is None:
        _config_cache.clear()
    else:
        keys_to_remove = [k for k in _config_cache if k.endswith(f":{config_name}")]
        for key in keys_to_remove:
            del _config_cache[key]


class Settings:
    """
    Typed access to settings configuration.

    Provides attribute-based access with type hints.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self._config = load_config('settings', base_dir)

    @property
    def timezone(self) -> str:
        return self._config['meta']['timezone']

    @property
    def asof_time(self) -> str:
        return self._config['runtime']['asof_time_et']

    @property
    def target_holdings(self) -> int:
        return self._config['portfolio']['target_holdings']

    @property
    def max_position_cap(self) -> float:
        return self._config['portfolio']['max_position_cap']

    @property
    def max_sector_cap(self) -> float:
        return self._config['portfolio']['max_sector_cap']

    @property
    def vol_target(self) -> float:
        return self._config['portfolio']['vol_target_annual']

    @property
    def artifacts_dir(self) -> str:
        return self._config['paths']['artifacts_dir']


class RiskLimits:
    """Typed access to risk limits configuration."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._config = load_config('risk_limits', base_dir)

    @property
    def drawdown_level_1(self) -> float:
        return self._config['drawdown']['level_1']['mdd_threshold']

    @property
    def drawdown_level_2(self) -> float:
        return self._config['drawdown']['level_2']['mdd_threshold']

    @property
    def kill_switch_threshold(self) -> float:
        return self._config['drawdown']['kill_switch']['mdd_threshold']

    @property
    def cooldown_days(self) -> int:
        return self._config['drawdown']['kill_switch']['cooldown_days']

    @property
    def monthly_turnover_cap(self) -> float:
        return self._config['turnover']['monthly_cap']


class GovernancePolicy:
    """Typed access to governance policy configuration."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._config = load_config('governance_policy', base_dir)

    @property
    def llm_enabled(self) -> bool:
        return self._config['llm_global']['enabled']

    @property
    def max_llm_weight_delta(self) -> float:
        return self._config['budgets']['llm']['max_total_weight_l1_delta']

    @property
    def actions_whitelist(self) -> list:
        return self._config['actions_whitelist']

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific LLM module."""
        return self._config['modules'].get(module_name, {})

    @property
    def severe_flag_overrides(self) -> list:
        return self._config['aggregation']['severe_flag_overrides']


def validate_all_configs(base_dir: Optional[Path] = None) -> Dict[str, bool]:
    """
    Validate all configuration files exist and are parseable.

    Returns:
        Dictionary mapping config names to validation status
    """
    config_names = [
        'settings', 'risk_limits', 'governance_policy',
        'factor_defs', 'universe_rules', 'execution_policy', 'monitoring'
    ]

    results = {}
    for name in config_names:
        try:
            load_config(name, base_dir)
            results[name] = True
        except Exception as e:
            results[name] = False
            print(f"Config validation failed for {name}: {e}")

    return results
