from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import ClientBrand, UserClient, get_db
from services.auth_service import get_current_user
from services.request_context import current_request_method

router = APIRouter()

# H6: same role hierarchy as scans._check_scan_access
_ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}
_DESTRUCTIVE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class BrandCreate(BaseModel):
    name: str
    parent_id: str | None = None
    domain: str | None = None
    aliases: list[str] | None = None


class BrandUpdate(BaseModel):
    name: str | None = None
    parent_id: str | None = None
    domain: str | None = None
    aliases: list[str] | None = None


def _check_client_access(client_id: str, user, db: Session):
    """Thin wrapper over services.access.check_client_access (Phase E.C).

    Kept under the legacy name so existing call sites don't need to change.
    """
    from services.access import check_client_access
    check_client_access(client_id, user, db)


def _serialize_brand(b: ClientBrand) -> dict:
    return {
        "id": str(b.id),
        "parent_id": str(b.parent_id) if b.parent_id else None,
        "name": b.name,
        "aliases": b.aliases or [],
        "domain": b.domain,
        "detection_source": b.detection_source,
        "auto_detected": b.auto_detected,
        "validated_by_user": b.validated_by_user,
        "first_detected_at": b.first_detected_at.isoformat() if b.first_detected_at else None,
    }


@router.get("/{client_id}/brands")
async def list_brands(
    client_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all brands detected for a client (catalog view)."""
    _check_client_access(client_id, user, db)
    brands = db.query(ClientBrand).filter(
        ClientBrand.client_id == client_id,
    ).order_by(ClientBrand.name).all()
    return [_serialize_brand(b) for b in brands]


@router.post("/{client_id}/brands")
async def create_brand(client_id: str, req: BrandCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_client_access(client_id, user, db)

    existing = db.query(ClientBrand).filter(
        ClientBrand.client_id == client_id, ClientBrand.name == req.name,
    ).first()
    if existing:
        raise HTTPException(400, f"Brand '{req.name}' already exists")

    brand = ClientBrand(
        client_id=client_id,
        parent_id=req.parent_id,
        name=req.name,
        domain=req.domain,
        aliases=req.aliases or [],
        detection_source="manual",
        auto_detected=False,
        validated_by_user=True,
    )
    db.add(brand)
    db.commit()
    db.refresh(brand)
    return _serialize_brand(brand)


@router.patch("/{client_id}/brands/{brand_id}")
async def update_brand(client_id: str, brand_id: str, req: BrandUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_client_access(client_id, user, db)
    brand = db.query(ClientBrand).filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id).first()
    if not brand:
        raise HTTPException(404, "Brand not found")

    if req.name is not None:
        brand.name = req.name
    if req.parent_id is not None:
        brand.parent_id = req.parent_id if req.parent_id != "" else None
    if req.domain is not None:
        brand.domain = req.domain
    if req.aliases is not None:
        brand.aliases = req.aliases
    db.commit()
    return _serialize_brand(brand)


@router.delete("/{client_id}/brands/{brand_id}")
async def delete_brand(client_id: str, brand_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    _check_client_access(client_id, user, db)
    brand = db.query(ClientBrand).filter(ClientBrand.id == brand_id, ClientBrand.client_id == client_id).first()
    if not brand:
        raise HTTPException(404, "Brand not found")
    db.delete(brand)
    db.commit()
    return {"deleted": True}
