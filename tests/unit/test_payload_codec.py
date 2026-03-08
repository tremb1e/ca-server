from src.crypto.payload_codec import decrypt_then_decompress


class _DecryptorOk:
    def __init__(self, plaintext: bytes):
        self.plaintext = plaintext

    def decrypt(self, encrypted_data: bytes):
        return True, self.plaintext, None


class _DecryptorFail:
    def decrypt(self, encrypted_data: bytes):
        return False, None, "decrypt_error"


class _DecompressorSpy:
    def __init__(self, output: bytes = b"", success: bool = True):
        self.called = False
        self.last_input = None
        self.last_hint = None
        self.output = output
        self.success = success

    def decompress(self, compressed_data: bytes, compression_hint=None):
        self.called = True
        self.last_input = compressed_data
        self.last_hint = compression_hint
        if self.success:
            return True, self.output, None
        return False, None, "decompress_error"


def test_decrypt_then_decompress_order_and_data_flow():
    decryptor = _DecryptorOk(plaintext=b"decrypted_bytes")
    decompressor = _DecompressorSpy(output=b"final_payload", success=True)

    success, payload, reason, error = decrypt_then_decompress(
        encrypted_payload=b"encrypted_bytes",
        decryptor=decryptor,
        decompressor=decompressor,
        compression_hint="gzip",
    )

    assert success is True
    assert payload == b"final_payload"
    assert reason is None
    assert error is None
    assert decompressor.called is True
    assert decompressor.last_input == b"decrypted_bytes"
    assert decompressor.last_hint == "gzip"


def test_decrypt_failure_stops_before_decompress():
    decryptor = _DecryptorFail()
    decompressor = _DecompressorSpy(output=b"unused", success=True)

    success, payload, reason, error = decrypt_then_decompress(
        encrypted_payload=b"encrypted_bytes",
        decryptor=decryptor,
        decompressor=decompressor,
    )

    assert success is False
    assert payload is None
    assert reason == "decryption_failed"
    assert error == "decrypt_error"
    assert decompressor.called is False


def test_decompress_failure_returns_decompression_error():
    decryptor = _DecryptorOk(plaintext=b"decrypted_bytes")
    decompressor = _DecompressorSpy(output=b"", success=False)

    success, payload, reason, error = decrypt_then_decompress(
        encrypted_payload=b"encrypted_bytes",
        decryptor=decryptor,
        decompressor=decompressor,
    )

    assert success is False
    assert payload is None
    assert reason == "decompression_failed"
    assert error == "decompress_error"
    assert decompressor.called is True
