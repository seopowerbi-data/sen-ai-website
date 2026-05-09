from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import Client, ClientBrand, ClientCredit, ScanBrandClassification, UserClient, get_db
from services.auth_service import get_current_user
from services.request_context import current_request_method
from services.sanitize import strip_tags

router = APIRouter()

# RBAC mirror of brands.py — viewer can read, editor+ can write
_ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}
_DESTRUCTIVE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _check_client_access(client_id: str, user, db: Session):
    """Same role-gate pattern as brands.py:_check_client_access."""
    link = db.query(UserClient).filter(
        UserClient.user_id == user.id, UserClient.client_id == client_id,
    ).first()
    if not link:
        raise HTTPException(403, "Access denied")
    method = current_request_method.get()
    if method in _DESTRUCTIVE_METHODS:
        rank = _ROLE_RANK.get(link.role, -1)
        if rank < _ROLE_RANK["editor"]:
            raise HTTPException(
                403,
                f"Insufficient role: '{link.role}' cannot {method} client settings "
                f"(requires 'editor' or 'owner')",
            )


class ClientResponse(BaseModel):
    id: str
    name: str
    brand: str | None
    apps: dict | None = None

    model_config = {"from_attributes": True}


class ClientCreate(BaseModel):
    name: str
    brand: str | None = None


@router.get("/", response_model=list[ClientResponse])
async def list_clients(user=Depends(get_current_user), db: Session = Depends(get_db)):
    links = db.query(UserClient).filter(UserClient.user_id == user.id).all()
    client_ids = [link.client_id for link in links]
    clients = db.query(Client).filter(Client.id.in_(client_ids)).all()
    return [ClientResponse(id=str(c.id), name=c.name, brand=c.brand, apps=c.apps) for c in clients]


@router.post("/")
async def create_client(req: ClientCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if user already has a client
    existing = db.query(UserClient).filter(UserClient.user_id == user.id).first()
    if existing:
        client = db.query(Client).filter(Client.id == existing.client_id).first()
        return ClientResponse(id=str(client.id), name=client.name, brand=client.brand, apps=client.apps)

    # Create new client + link user as owner
    # Welcome bonus is now granted on email verification (H3), not here
    client = Client(name=strip_tags(req.name), brand=strip_tags(req.brand))
    db.add(client)
    db.flush()

    db.add(UserClient(user_id=user.id, client_id=client.id, role="owner"))

    db.commit()
    db.refresh(client)

    return ClientResponse(id=str(client.id), name=client.name, brand=client.brand, apps=client.apps)


# ── Brand promotion settings ────────────────────────────────────────────
# These endpoints back the Workspace Settings → "My primary brands" UI.
# primary_brand_ids is the cross-scan default for content-gen promotion
# (FAQ / Article generation). Resolution chain documented in
# worker/services/brand_resolver.py.

class PromotionUpdate(BaseModel):
    primary_brand_ids: list[str]  # ordered, [0] = lead brand


@router.get("/{client_id}/promotion")
async def get_client_promotion(client_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Return current promotion settings + all client brands + auto-detected suggestions."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    primary_ids = list(client.primary_brand_ids or [])
    primary_id_set = {str(bid) for bid in primary_ids}

    all_brands = (
        db.query(ClientBrand)
        .filter(ClientBrand.client_id == client_id)
        .order_by(ClientBrand.name)
        .all()
    )

    # Auto-detected my_brand brands (across all scans for this client)
    suggested_rows = (
        db.query(ScanBrandClassification.brand_id)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(
            ClientBrand.client_id == client_id,
            ScanBrandClassification.classification == "my_brand",
        )
        .distinct()
        .all()
    )
    suggested_id_set = {str(r.brand_id) for r in suggested_rows}

    by_id = {str(b.id): b for b in all_brands}
    serialized: list[dict] = []
    for bid in primary_ids:
        bid_str = str(bid)
        b = by_id.get(bid_str)
        if b:
            serialized.append({
                "id": bid_str, "name": b.name, "domain": b.domain,
                "is_primary": True, "is_suggested": bid_str in suggested_id_set,
            })
    for b in all_brands:
        bid_str = str(b.id)
        if bid_str in primary_id_set:
            continue
        serialized.append({
            "id": bid_str, "name": b.name, "domain": b.domain,
            "is_primary": False, "is_suggested": bid_str in suggested_id_set,
        })

    return {
        "primary_brand_ids": [str(bid) for bid in primary_ids],
        "all_brands": serialized,
    }


@router.put("/{client_id}/promotion")
async def update_client_promotion(client_id: str, req: PromotionUpdate,
                                  user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Replace the client's primary_brand_ids (workspace default for content gen)."""
    _check_client_access(client_id, user, db)
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(404, "Client not found")

    valid_ids = {
        str(b.id) for b in
        db.query(ClientBrand).filter(ClientBrand.client_id == client_id).all()
    }
    invalid = [bid for bid in req.primary_brand_ids if bid not in valid_ids]
    if invalid:
        raise HTTPException(400, f"Brand IDs not in this client: {invalid[:3]}")

    try:
        client.primary_brand_ids = [UUID(bid) for bid in req.primary_brand_ids]
    except ValueError as e:
        raise HTTPException(400, f"Malformed UUID: {e}")
    db.commit()

    return {
        "ok": True,
        "primary_brand_ids": req.primary_brand_ids,
        "count": len(req.primary_brand_ids),
    }
