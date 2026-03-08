from typing import Optional, Protocol, Tuple

class Decryptor(Protocol):
    def decrypt(self, encrypted_data: bytes) -> Tuple[bool, Optional[bytes], Optional[str]]:
        ...


class Decompressor(Protocol):
    def decompress(
        self,
        compressed_data: bytes,
        compression_hint: Optional[str] = None,
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        ...


def decrypt_then_decompress(
    encrypted_payload: bytes,
    decryptor: Decryptor,
    decompressor: Decompressor,
    compression_hint: Optional[str] = None,
) -> Tuple[bool, Optional[bytes], Optional[str], Optional[str]]:
    """Process inbound payload with a strict decrypt -> decompress order."""
    decrypt_ok, decrypted_payload, decrypt_err = decryptor.decrypt(encrypted_payload)
    if not decrypt_ok or decrypted_payload is None:
        return False, None, "decryption_failed", decrypt_err

    decompress_ok, decompressed_payload, decompress_err = decompressor.decompress(
        decrypted_payload,
        compression_hint=compression_hint,
    )
    if not decompress_ok or decompressed_payload is None:
        return False, None, "decompression_failed", decompress_err

    return True, decompressed_payload, None, None
