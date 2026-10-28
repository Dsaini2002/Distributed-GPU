import hashlib
import hmac
import os
import secrets
import time

from fastapi import APIRouter, Request
from fastapi.responses import (
    HTMLResponse, JSONResponse, RedirectResponse
)
from pydantic import BaseModel

router = APIRouter()
COOKIE = "dgpu_session"
SESSION_SECONDS = 12 * 60 * 60


def secret_key():
    token = os.environ.get("DGPU_TOKEN")
    if not token:
        raise RuntimeError("DGPU_TOKEN not configured")

    return hashlib.sha256(
        ("dgpu-session-v1:" + token).encode()
    ).digest()


def make_session(key, now=None):
    now = int(time.time() if now is None else now)

    message = (
        str(now + SESSION_SECONDS)
        + "."
        + secrets.token_urlsafe(16)
    )

    signature = hmac.new(
        key, message.encode(), hashlib.sha256
    ).hexdigest()

    return message + "." + signature


def valid_session(value, key, now=None):
    if not value or len(value) > 300:
        return False

    parts = value.rsplit(".", 1)
    if len(parts) != 2:
        return False

    message, signature = parts

    try:
        expiry = int(message.split(".", 1)[0])
    except ValueError:
        return False

    current = int(time.time() if now is None else now)

    if not current < expiry <= current + SESSION_SECONDS:
        return False

    expected = hmac.new(
        key, message.encode(), hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected)


class LoginData(BaseModel):
    token: str


@router.get("/login", response_class=HTMLResponse)
def login_page():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distributed-GPU Login</title>
<style>
body {
  background:#101827;color:#eef5ff;
  font:16px Arial;display:grid;place-items:center;
  min-height:90vh;
}
main {
  background:#1c2940;padding:32px;border-radius:16px;
  width:min(90%,420px);
}
h1 {color:#77baff}
input,button {
  box-sizing:border-box;width:100%;padding:13px;
  border-radius:8px;margin:10px 0;
}
button {background:#318ce7;color:white;border:0}
small {color:#a9bdd4}
</style>
</head>
<body>
<main>
<h1>Distributed-GPU Studio</h1>
<p>Cluster Administrator Login</p>
<input id="secret" type="password"
 placeholder="Enter cluster token" autocomplete="off">
<button onclick="login()">Login</button>
<p id="message"></p>
<small>Private trusted LAN prototype</small>
</main>
<script>
async function login() {
  const message = document.getElementById("message");
  const input = document.getElementById("secret");

  try {
    const response = await fetch("/api/session/login",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      credentials:"same-origin",
      body:JSON.stringify({token:input.value})
    });

    input.value = "";

    if (!response.ok) {
      message.textContent = "Invalid token or login limit reached";
      return;
    }

    location.replace("/train");
  } catch(error) {
    message.textContent = "Connection failed";
  }
}
document.getElementById("secret").addEventListener(
  "keydown", e => {if(e.key==="Enter") login()}
);
</script>
</body>
</html>
"""


# Simple in-memory login rate limiting
failures = {}


@router.post("/api/session/login")
def login(data: LoginData, request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()

    recent = [
        t for t in failures.get(ip, [])
        if now - t < 300
    ]

    if len(recent) >= 5:
        return JSONResponse(
            {"detail": "Too many attempts"},
            status_code=429
        )

    expected = os.environ.get("DGPU_TOKEN", "")

    if not expected or not hmac.compare_digest(
        data.token, expected
    ):
        recent.append(now)
        failures[ip] = recent
        return JSONResponse(
            {"detail": "Invalid token"},
            status_code=401
        )

    failures.pop(ip, None)

    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        COOKIE,
        make_session(secret_key()),
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/"
    )
    return response


@router.post("/api/session/logout")
def logout():
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(COOKIE, path="/")
    return response


def install(app):
    app.include_router(router)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        path = request.url.path

        if path in ("/login", "/api/session/login"):
            return await call_next(request)

        # Agent registration already validates its bearer token.
        if path == "/api/heartbeat":
            return await call_next(request)

        token = os.environ.get("DGPU_TOKEN", "")
        header = request.headers.get("authorization", "")
        valid_bearer = bool(token) and hmac.compare_digest(
            header, "Bearer " + token
        )

        authenticated = valid_session(
            request.cookies.get(COOKIE),
            secret_key()
        )

        if authenticated:
            # Basic same-origin protection for cookie requests.
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                origin = request.headers.get("origin")
                expected_origin = (
                    request.url.scheme + "://"
                    + request.headers.get("host", "")
                )

                if origin and origin != expected_origin:
                    return JSONResponse(
                        {"detail": "Invalid origin"},
                        status_code=403
                    )

            # Existing APIs still expect Authorization header.
            # Supply it internally after session verification.
            headers = [
                (k, v) for k, v in request.scope["headers"]
                if k.lower() != b"authorization"
            ]

            headers.append((
                b"authorization",
                ("Bearer " + token).encode()
            ))

            request.scope["headers"] = headers
            return await call_next(request)

        if valid_bearer:
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse(
                {"detail": "Login required"},
                status_code=401
            )

        return RedirectResponse("/login", status_code=303)
