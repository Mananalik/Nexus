# services/clerk_auth.py
import os
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

# HTTPBearer used only if you want to accept Authorization: Bearer <token> directly.
# We'll accept either cookie (__session) or Authorization header via authenticate_request.
security = HTTPBearer(auto_error=False)

# Set this env var to your Clerk Secret Key (Dashboard -> API Keys)
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")  # e.g. sk_xxx

# Optional: restrict to your frontend origin(s) or frontend API (authorized parties)
AUTHORIZED_PARTIES = os.getenv("CLERK_AUTHORIZED_PARTIES", "")  # comma-separated if multiple

# Build authenticate options (Authorized parties if provided)
def _build_auth_options():
    ap = [p.strip() for p in AUTHORIZED_PARTIES.split(",") if p.strip()]
    return AuthenticateRequestOptions(authorized_parties=ap) if ap else AuthenticateRequestOptions()

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """
    FastAPI dependency that:
    - Uses the Clerk Python SDK to authenticate the incoming request (cookie or Authorization header).
    - Returns a dict with clerk_user_id, email, name and the whole payload.
    Raises HTTPException(401) on failure.
    """
    if not CLERK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CLERK_SECRET_KEY not configured on the server")

    # Initialize SDK client with your secret key
    # Use sync client context - SDK calls used below are synchronous (authenticate_request is sync)
    sdk = Clerk(bearer_auth=CLERK_SECRET_KEY)

    # Build an httpx.Request that mirrors the incoming FastAPI request headers/cookies.
    # authenticate_request reads cookies and Authorization header.
    # We can pass a dummy URL; authenticate_request only needs headers/cookies present.
    headers = {k.decode(): v.decode() for k, v in request.scope["headers"]}
    # If Authorization was provided via HTTPBearer (credentials), ensure header is present
    if credentials and credentials.credentials:
        headers["authorization"] = f"Bearer {credentials.credentials}"

    httpx_req = httpx.Request("GET", "https://backend-placeholder.local", headers=headers)

    # Authenticate the request using the SDK helper
    try:
        request_state = sdk.authenticate_request(httpx_req, _build_auth_options())
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Clerk authentication error: {e}")

    if not request_state.is_signed_in:
        # request_state.reason contains the failure reason (useful for debugging)
        raise HTTPException(status_code=401, detail=f"Not authenticated: {getattr(request_state, 'reason', 'unknown')}")

    # payload contains the verified token/session claims
    payload = getattr(request_state, "payload", {}) or {}
    clerk_user_id = payload.get("sub") or payload.get("user_id")
    email = payload.get("email") or payload.get("email_address")
    name = payload.get("name") or payload.get("given_name")

    return {"clerk_user_id": clerk_user_id, "email": email, "name": name, "payload": payload}
