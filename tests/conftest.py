import pytest


@pytest.fixture(autouse=True)
def _http_test_client_cookies(monkeypatch):
    """Tests drive the app over plain http://test, and an http client does not
    send back a Secure cookie. Tests of the flag itself clear this."""
    monkeypatch.setenv("R3_INSECURE_COOKIES", "1")
