# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from app.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    clerk_user_id = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, index=True, nullable=True)
    full_name = Column(String, nullable=True)
    metadata_json = Column(JSON, default={})   
    created_at = Column(DateTime(timezone=True), server_default=func.now())
