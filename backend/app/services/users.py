# app/services/users.py
from sqlalchemy import select
from app.models import User

async def get_or_create_user_from_clerk(db, clerk_info: dict):
    clerk_id = clerk_info.get("clerk_user_id")
    if not clerk_id:
        raise ValueError("Missing clerk_user_id")
    q = select(User).where(User.clerk_user_id == clerk_id)
    res = await db.execute(q)
    user = res.scalars().first()
    if user:
        # optional: update email/name if changed
        updated = False
        if clerk_info.get("email") and user.email != clerk_info.get("email"):
            user.email = clerk_info["email"]; updated = True
        if clerk_info.get("name") and user.full_name != clerk_info.get("name"):
            user.full_name = clerk_info["name"]; updated = True
        if updated:
            await db.commit()
            await db.refresh(user)
        return user
    user = User(clerk_user_id=clerk_id, email=clerk_info.get("email"), full_name=clerk_info.get("name"))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
