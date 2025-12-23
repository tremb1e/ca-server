import pytest
import gzip

import lz4.frame

from src.crypto.decompression import DataDecompressor


class TestDataDecompressor:
    @pytest.fixture
    def decompressor(self):
        return DataDecompressor()

    @pytest.fixture
    def test_data(self):
        original = b"Hello, World! " * 100
        compressed = lz4.frame.compress(original)
        return {
            "original": original,
            "compressed": compressed
        }

    def test_successful_lz4_decompression(self, decompressor, test_data):
        success, decompressed, error = decompressor.decompress(test_data["compressed"], "lz4")

        assert success is True
        assert error is None
        assert decompressed == test_data["original"]

    def test_successful_gzip_decompression(self, decompressor):
        original = b"gzip payload" * 50
        compressed = gzip.compress(original)

        success, decompressed, error = decompressor.decompress(compressed, "gzip")

        assert success is True
        assert error is None
        assert decompressed == original

    def test_auto_detect_gzip(self, decompressor):
        original = b"gzip detection" * 10
        compressed = gzip.compress(original)

        success, decompressed, error = decompressor.decompress(compressed)

        assert success is True
        assert error is None
        assert decompressed == original

    def test_decompress_empty_data(self, decompressor):
        success, decompressed, error = decompressor.decompress(b"")

        assert success is False
        assert decompressed is None
        assert "empty" in error.lower()

    def test_decompress_invalid_data(self, decompressor):
        invalid_data = b"not compressed data"
        success, decompressed, error = decompressor.decompress(invalid_data)

        assert success is False
        assert decompressed is None
        assert error is not None
        assert len(error) > 0

    def test_decompress_small_data(self, decompressor):
        original = b"Small"
        compressed = lz4.frame.compress(original)
        success, decompressed, error = decompressor.decompress(compressed)

        assert success is True
        assert error is None
        assert decompressed == original

    def test_decompress_large_data(self, decompressor):
        original = b"x" * 1000000
        compressed = lz4.frame.compress(original)
        success, decompressed, error = decompressor.decompress(compressed)

        assert success is True
        assert error is None
        assert decompressed == original
        assert len(decompressed) == 1000000
