"""
data/database.py
================
Couche base de données SQLite (SQLAlchemy).

Schéma : 2 tables liées par calc_id
─────────────────────────────────────
TABLE inputs
  calc_id     TEXT  PRIMARY KEY   (UUID court)
  timestamp   TEXT
  style       TEXT  ('european' | 'american')
  option_type TEXT  ('call' | 'put')
  S           REAL  prix sous-jacent
  K           REAL  strike
  t           REAL  maturité (années)
  r           REAL  taux sans risque
  sigma       REAL  volatilité
  y           REAL  dividende
  entry_price REAL  prix d'entrée payé (pour P&L)
  method      TEXT  méthode de pricing utilisée

TABLE outputs
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  calc_id     TEXT    FK → inputs.calc_id
  shock_S     REAL    choc en % sur S   (ex: -0.20 = -20%)
  shock_sigma REAL    choc en % sur σ   (ex: +0.10 = +10 vol pts)
  S_shocked   REAL    valeur absolue de S après choc
  sigma_shocked REAL  valeur absolue de σ après choc
  option_price REAL   prix de l'option pour ce scénario
  pnl         REAL    option_price - entry_price
"""

import os
import uuid
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Text, Integer, Float,
    ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Chemin DB ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(_BASE_DIR, "pricer.db")
DB_URL    = f"sqlite:///{DB_PATH}"

Base   = declarative_base()
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# Modèles ORM
# ─────────────────────────────────────────────────────────────────────────────

class InputRecord(Base):
    __tablename__ = "inputs"

    calc_id     = Column(Text,    primary_key=True)
    timestamp   = Column(Text,    nullable=False)
    style       = Column(Text,    nullable=False)
    option_type = Column(Text,    nullable=False)
    S           = Column(Float,   nullable=False)
    K           = Column(Float,   nullable=False)
    t           = Column(Float,   nullable=False)
    r           = Column(Float,   nullable=False)
    sigma       = Column(Float,   nullable=False)
    y           = Column(Float,   nullable=False, default=0.0)
    entry_price = Column(Float,   nullable=True)   # None si non renseigné
    method      = Column(Text,    nullable=False, default="BlackScholes")

    outputs = relationship("OutputRecord", back_populates="input",
                           cascade="all, delete-orphan")

    def to_dict(self):
        return {c.name: getattr(self, c.name)
                for c in self.__table__.columns}


class OutputRecord(Base):
    __tablename__ = "outputs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    calc_id       = Column(Text,  ForeignKey("inputs.calc_id"), nullable=False)
    shock_S       = Column(Float, nullable=False)   # % de choc sur S
    shock_sigma   = Column(Float, nullable=False)   # choc absolu sur σ
    S_shocked     = Column(Float, nullable=False)
    sigma_shocked = Column(Float, nullable=False)
    option_price  = Column(Float, nullable=False)
    pnl           = Column(Float, nullable=True)    # None si entry_price absent

    input = relationship("InputRecord", back_populates="outputs")

    def to_dict(self):
        return {c.name: getattr(self, c.name)
                for c in self.__table__.columns}


Index("ix_outputs_calc_id", OutputRecord.calc_id)

# Création des tables au premier import
Base.metadata.create_all(engine)


# ─────────────────────────────────────────────────────────────────────────────
# Context manager session
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_session():
    s = Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────

def save_calculation(
    style: str, option_type: str,
    S: float, K: float, t: float, r: float, sigma: float, y: float,
    entry_price: float | None,
    method: str,
    shock_results: list[dict],       # liste de dicts {shock_S, shock_sigma, option_price}
) -> str:
    """
    Enregistre un calcul complet (1 ligne inputs + N lignes outputs).

    shock_results : liste de dicts avec clés
        shock_S, shock_sigma, S_shocked, sigma_shocked, option_price

    Retourne le calc_id.
    """
    calc_id = str(uuid.uuid4())[:8]
    ts      = datetime.utcnow().isoformat(timespec="seconds")

    with get_session() as s:
        inp = InputRecord(
            calc_id=calc_id, timestamp=ts,
            style=style, option_type=option_type,
            S=S, K=K, t=t, r=r, sigma=sigma, y=y,
            entry_price=entry_price, method=method,
        )
        s.add(inp)

        for row in shock_results:
            ep  = entry_price or 0.0
            pnl = row["option_price"] - ep if entry_price is not None else None
            s.add(OutputRecord(
                calc_id       = calc_id,
                shock_S       = row["shock_S"],
                shock_sigma   = row["shock_sigma"],
                S_shocked     = row["S_shocked"],
                sigma_shocked = row["sigma_shocked"],
                option_price  = row["option_price"],
                pnl           = pnl,
            ))

    return calc_id


def get_calculation(calc_id: str) -> dict | None:
    """Récupère inputs + outputs d'un calc_id."""
    with get_session() as s:
        inp = s.query(InputRecord).filter_by(calc_id=calc_id).first()
        if inp is None:
            return None
        return {
            "inputs":  inp.to_dict(),
            "outputs": [o.to_dict() for o in inp.outputs],
        }


def list_calculations(limit: int = 50) -> list[dict]:
    """Liste les derniers calculs (inputs uniquement)."""
    with get_session() as s:
        rows = (s.query(InputRecord)
                 .order_by(InputRecord.timestamp.desc())
                 .limit(limit)
                 .all())
        return [r.to_dict() for r in rows]


def delete_calculation(calc_id: str) -> bool:
    with get_session() as s:
        inp = s.query(InputRecord).filter_by(calc_id=calc_id).first()
        if inp is None:
            return False
        s.delete(inp)
    return True


def clear_all():
    with get_session() as s:
        s.query(OutputRecord).delete()
        s.query(InputRecord).delete()