"""
Tests for PIT Dataset Builder.

Verifies that:
1. Datasets are properly versioned and manifested
2. PIT filtering works correctly (no future data)
3. Snapshots are deterministic
4. Save/load preserves data integrity
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from alpha_research.data.pit_dataset import (
    PITDatasetBuilder,
    PITDataset,
    PITSnapshot,
    DataManifest,
    DataQuality,
    DataSource,
)


class TestPITDatasetBuilder:
    """Tests for PITDatasetBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a builder for testing."""
        return PITDatasetBuilder()

    def test_build_synthetic_dataset(self, builder):
        """Test building a synthetic dataset."""
        dataset = builder.build(
            universe="sp500_sample",
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            quality_level=DataQuality.SYNTHETIC,
        )

        # Verify manifest
        assert dataset.manifest is not None
        assert dataset.manifest.quality_level == "synthetic"
        assert dataset.manifest.n_symbols > 0
        assert dataset.manifest.n_trading_days > 0
        assert dataset.manifest.pit_validated is True

        # Verify prices
        assert len(dataset.prices) > 0
        assert 'symbol' in dataset.prices.columns
        assert 'trade_date' in dataset.prices.columns
        assert 'close' in dataset.prices.columns

    def test_dataset_has_hash(self, builder):
        """Test that datasets have content hashes."""
        dataset = builder.build(
            universe="sp500_sample",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 6, 30),
            quality_level=DataQuality.SYNTHETIC,
        )

        assert dataset.manifest.dataset_hash is not None
        assert len(dataset.manifest.dataset_hash) == 16
        assert dataset.manifest.prices_hash is not None

    def test_same_params_same_hash(self, builder):
        """Test that same parameters produce same hash (deterministic)."""
        dataset1 = builder.build(
            universe="sp500_sample",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 3, 31),
            quality_level=DataQuality.SYNTHETIC,
        )

        dataset2 = builder.build(
            universe="sp500_sample",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 3, 31),
            quality_level=DataQuality.SYNTHETIC,
        )

        # Hashes should be identical (deterministic)
        assert dataset1.manifest.prices_hash == dataset2.manifest.prices_hash


class TestPITDatasetSnapshot:
    """Tests for PIT snapshot functionality."""

    @pytest.fixture
    def dataset(self):
        """Create a dataset for testing."""
        builder = PITDatasetBuilder()
        return builder.build(
            universe="sp500_sample",
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            include_fundamentals=True,
            quality_level=DataQuality.SYNTHETIC,
        )

    def test_snapshot_excludes_future_data(self, dataset):
        """Test that snapshots exclude future data."""
        as_of = date(2022, 7, 1)
        snapshot = dataset.get_snapshot(as_of)

        # All price dates should be before as_of
        price_dates = pd.to_datetime(snapshot.prices['trade_date']).dt.date
        assert all(d < as_of for d in price_dates), "Snapshot contains future price data"

    def test_snapshot_excludes_future_fundamentals(self, dataset):
        """Test that fundamental snapshots exclude future data."""
        as_of = date(2022, 7, 1)
        snapshot = dataset.get_snapshot(as_of)

        if snapshot.fundamentals is not None and len(snapshot.fundamentals) > 0:
            available_dates = pd.to_datetime(snapshot.fundamentals['available_at']).dt.date
            assert all(d < as_of for d in available_dates), "Snapshot contains future fundamental data"

    def test_different_as_of_different_snapshot(self, dataset):
        """Test that different as_of dates produce different snapshots."""
        snapshot1 = dataset.get_snapshot(date(2022, 6, 1))
        snapshot2 = dataset.get_snapshot(date(2022, 12, 1))

        # Second snapshot should have more data
        assert len(snapshot2.prices) > len(snapshot1.prices)


class TestPITDatasetPersistence:
    """Tests for dataset save/load functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dataset(self):
        """Create a dataset for testing."""
        builder = PITDatasetBuilder()
        return builder.build(
            universe="sp500_sample",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 6, 30),
            include_fundamentals=True,
            quality_level=DataQuality.SYNTHETIC,
        )

    def test_save_and_load(self, dataset, temp_dir):
        """Test that save and load preserves data."""
        # Save
        dataset_dir = dataset.save(temp_dir)

        # Load
        loaded = PITDataset.load(dataset_dir)

        # Verify manifest
        assert loaded.manifest.dataset_id == dataset.manifest.dataset_id
        assert loaded.manifest.dataset_hash == dataset.manifest.dataset_hash

        # Verify prices
        assert len(loaded.prices) == len(dataset.prices)
        assert list(loaded.prices.columns) == list(dataset.prices.columns)

    def test_manifest_json_roundtrip(self, dataset):
        """Test manifest JSON serialization."""
        json_str = dataset.manifest.to_json()
        loaded = DataManifest.from_json(json_str)

        assert loaded.dataset_id == dataset.manifest.dataset_id
        assert loaded.n_symbols == dataset.manifest.n_symbols
        assert loaded.prices_hash == dataset.manifest.prices_hash


class TestDataManifest:
    """Tests for DataManifest."""

    def test_manifest_has_required_fields(self):
        """Test manifest has all required fields."""
        manifest = DataManifest(
            dataset_id="test_dataset",
            dataset_hash="abc123",
            created_at=datetime.utcnow().isoformat(),
            universe="sp500",
            start_date="2022-01-01",
            end_date="2023-12-31",
            price_source="yahoo_finance",
            fundamental_source=None,
            event_source=None,
            quality_level="research",
            n_symbols=50,
            n_trading_days=500,
            n_price_records=25000,
            n_fundamental_records=0,
            n_event_records=0,
            price_coverage_pct=0.95,
            fundamental_coverage_pct=0.0,
            pit_validated=True,
            pit_violations=0,
            prices_hash="def456",
            fundamentals_hash=None,
            events_hash=None,
        )

        assert manifest.dataset_id == "test_dataset"
        assert manifest.pit_validated is True
        assert manifest.n_symbols == 50

    def test_manifest_to_dict(self):
        """Test manifest to_dict conversion."""
        manifest = DataManifest(
            dataset_id="test",
            dataset_hash="hash",
            created_at="2024-01-01T00:00:00",
            universe="test",
            start_date="2022-01-01",
            end_date="2023-12-31",
            price_source="synthetic",
            fundamental_source=None,
            event_source=None,
            quality_level="synthetic",
            n_symbols=10,
            n_trading_days=100,
            n_price_records=1000,
            n_fundamental_records=0,
            n_event_records=0,
            price_coverage_pct=1.0,
            fundamental_coverage_pct=0.0,
            pit_validated=True,
            pit_violations=0,
            prices_hash="hash",
            fundamentals_hash=None,
            events_hash=None,
        )

        d = manifest.to_dict()
        assert isinstance(d, dict)
        assert d['dataset_id'] == 'test'
        assert d['n_symbols'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
