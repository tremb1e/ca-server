import pytest
from src.crypto.decryption import AESDecryptor
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
import os


class TestAESDecryptor:
    @pytest.fixture
    def decryptor(self):
        return AESDecryptor("Continuous_Authentication")

    @pytest.fixture
    def test_data(self):
        plaintext = b"Hello, World! This is a test message."
        password = "Continuous_Authentication"
        key = hashlib.sha256(password.encode('utf-8')).digest()

        iv = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        encrypted_data = iv + encryptor.tag + ciphertext

        return {
            "plaintext": plaintext,
            "encrypted": encrypted_data,
            "key": key
        }

    def test_successful_decryption(self, decryptor, test_data):
        success, decrypted, error = decryptor.decrypt(test_data["encrypted"])

        assert success is True
        assert error is None
        assert decrypted == test_data["plaintext"]

    def test_decrypt_empty_data(self, decryptor):
        success, decrypted, error = decryptor.decrypt(b"")

        assert success is False
        assert decrypted is None
        assert "too short" in error.lower()

    def test_decrypt_short_data(self, decryptor):
        success, decrypted, error = decryptor.decrypt(b"short")

        assert success is False
        assert decrypted is None
        assert "too short" in error.lower()

    def test_decrypt_invalid_data(self, decryptor):
        invalid_data = b"x" * 100
        success, decrypted, error = decryptor.decrypt(invalid_data)

        assert success is False
        assert decrypted is None
        assert "failed" in error.lower()

    def test_key_derivation(self, decryptor):
        password = "Continuous_Authentication"
        expected_key = hashlib.sha256(password.encode('utf-8')).digest()

        assert decryptor.key == expected_key
        assert len(decryptor.key) == 32