import unittest

from .session_auth import (
    make_session, valid_session
)


class TestSessionAuthentication(unittest.TestCase):
    def test_valid_session(self):
        key = b"test-secret"
        cookie = make_session(key, now=1000)
        self.assertTrue(valid_session(cookie, key, now=1001))

    def test_wrong_secret(self):
        cookie = make_session(b"correct", now=1000)
        self.assertFalse(
            valid_session(cookie, b"wrong", now=1001)
        )

    def test_expired_session(self):
        key = b"test-secret"
        cookie = make_session(key, now=1000)
        self.assertFalse(
            valid_session(cookie, key, now=50000)
        )


if __name__ == "__main__":
    unittest.main()
