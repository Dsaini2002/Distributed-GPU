"""Short-lived RSA-OAEP key wrapping for a controller token (never plaintext in HTTP)."""
import base64
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding


def new_pairing_key():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private, base64.b64encode(public_der).decode('ascii'), fingerprint(public_der)


def fingerprint(public_der: bytes):
    return hashlib.sha256(public_der).hexdigest()[:12].upper()


def parse_public_key(public_b64: str):
    if not isinstance(public_b64, str) or len(public_b64) > 2500:
        raise ValueError('Invalid public key length')
    try:
        der = base64.b64decode(public_b64, validate=True)
        key = serialization.load_der_public_key(der)
    except Exception as exc:
        raise ValueError('Bad public key') from exc
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size != 2048 or key.public_numbers().e != 65537:
        raise ValueError('Only RSA-2048 with e=65537 is supported')
    return key, fingerprint(der)


def wrap_token(key: rsa.RSAPublicKey, token: str):
    data = token.encode('utf-8')
    if not (5 <= len(data) <= 160):
        raise ValueError('Unexpected secret size')
    encrypted = key.encrypt(data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=b'DGPU-P20'))
    return base64.b64encode(encrypted).decode('ascii')


def unwrap_token(private: rsa.RSAPrivateKey, ciphertext: str):
    data = base64.b64decode(ciphertext, validate=True)
    return private.decrypt(data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=b'DGPU-P20')).decode('utf-8')

