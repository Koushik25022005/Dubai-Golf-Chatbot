import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

DB_FILE = os.path.join(os.path.dirname(__file__), "admins.db")
engine = create_engine(f"sqlite:///{DB_FILE}")
Base = declarative_base()

class Admin(Base):
    __tablename__ = 'admins'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def initialize_admins():
    db = SessionLocal()
    # Check if admins exist
    if db.query(Admin).count() == 0:
        koushik = Admin(username="Koushik", password="Kadmin123")
        yahya = Admin(username="Yahya", password="yadmin123")
        db.add(koushik)
        db.add(yahya)
        db.commit()
    db.close()

def authenticate_admin(username, password):
    db = SessionLocal()
    admin = db.query(Admin).filter(Admin.username == username, Admin.password == password).first()
    db.close()
    return admin is not None

# Initialize default users immediately when imported
initialize_admins()
