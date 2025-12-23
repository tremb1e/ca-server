import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AESDecryptor:
    def __init__(self, key: str):
        self.key = self._derive_key(key)
        logger.info("AESDecryptor initialized with derived key")

    def _derive_key(self, password: str) -> bytes:
        key = hashlib.sha256(password.encode('utf-8')).digest()
        return key

    def decrypt(self, encrypted_data: bytes) -> Tuple[bool, Optional[bytes], Optional[str]]:
        try:
            if len(encrypted_data) < 28:
                error_msg = f"Encrypted data too short: {len(encrypted_data)} bytes"
                logger.error(error_msg)
                return False, None, error_msg

            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]

            logger.debug(f"Decrypting data - IV length: {len(iv)}, Tag length: {len(tag)}, Ciphertext length: {len(ciphertext)}")

            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            logger.info(f"Successfully decrypted {len(encrypted_data)} bytes to {len(plaintext)} bytes")
            return True, plaintext, None

        except Exception as e:
            error_msg = f"Decryption failed: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg