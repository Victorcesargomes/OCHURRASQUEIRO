from __future__ import annotations

import logging
import re
import sqlite3
import atexit
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict, Optional, List, Dict, Any

import pandas as pd

# Verifica se estamos no Render (com disco persistente)
if os.environ.get("RENDER"):
    # Define o caminho do banco de dados no diretório persistente
    DB_PATH = Path("/data") / "financeiro.db"
    # Garante que o diretório existe
    Path("/data").mkdir(parents=True, exist_ok=True)
else:
    # Comportamento local
    BASE_PATH = Path(__file__).parent
    DB_PATH = BASE_PATH / "financeiro.db"

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS transacoes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    data      TEXT    NOT NULL,
    tipo      TEXT    NOT NULL CHECK (tipo IN ('faturamento','despesa')),
    valor     REAL    NOT NULL,
    descricao TEXT
);
"""

INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_tipo_data ON transacoes (tipo, data);"

class Registro(TypedDict):
    data: str
    tipo: Literal["faturamento", "despesa"]
    valor: float
    descricao: str | None

_conn_singleton: sqlite3.Connection | None = None

def _conn() -> sqlite3.Connection:
    global _conn_singleton
    if _conn_singleton is None:
        _conn_singleton = sqlite3.connect(
            str(DB_PATH),  # Convertendo para string para compatibilidade
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        _conn_singleton.row_factory = sqlite3.Row
        _conn_singleton.execute("PRAGMA journal_mode=WAL;")
        _conn_singleton.execute(DDL)
        _conn_singleton.execute(INDEX_SQL)
    return _conn_singleton

atexit.register(lambda: _conn_singleton and _conn_singleton.close())

def validar_registro(r: Registro) -> bool:
    """Valida os campos do registro antes de inserir."""
    try:
        # Validar data
        datetime.strptime(r["data"], "%Y-%m-%d")
        # Validar valor (deve ser positivo)
        if r["valor"] <= 0:
            return False
        # Validar tipo
        if r["tipo"] not in ("faturamento", "despesa"):
            return False
        return True
    except (ValueError, TypeError):
        return False

def inserir(r: Registro) -> None:
    if not validar_registro(r):
        raise ValueError("Registro inválido")
    c = _conn()
    c.execute(
        "INSERT INTO transacoes (data, tipo, valor, descricao) VALUES (?,?,?,?)",
        (r["data"], r["tipo"], r["valor"], r["descricao"]),
    )
    c.commit()

def deletar(reg_id: int) -> bool:
    c = _conn()
    cursor = c.execute("DELETE FROM transacoes WHERE id = ?", (reg_id,))
    c.commit()
    return cursor.rowcount > 0  # Retorna True se excluiu algum registro

def totais(
    data_inicio: Optional[str] = None, 
    data_fim: Optional[str] = None
) -> dict[str, float]:
    c = _conn()
    
    query = """
        SELECT tipo, SUM(valor) 
        FROM transacoes
    """
    
    conditions = []
    params = []
    
    if data_inicio:
        conditions.append("date(data) >= ?")
        params.append(data_inicio)
    if data_fim:
        conditions.append("date(data) <= ?")
        params.append(data_fim)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " GROUP BY tipo"
    
    return {
        t: v or 0.0
        for t, v in c.execute(query, params).fetchall()
    }

def listar_transacoes(
    data_inicio: Optional[str] = None, 
    data_fim: Optional[str] = None
) -> pd.DataFrame:
    c = _conn()
    
    query = """
        SELECT id, data, tipo, valor,
               COALESCE(descricao,'') AS descricao
        FROM transacoes
    """
    
    conditions = []
    params = []
    
    if data_inicio:
        conditions.append("date(data) >= ?")
        params.append(data_inicio)
    if data_fim:
        conditions.append("date(data) <= ?")
        params.append(data_fim)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY date(data) DESC, id DESC"
    
    return pd.read_sql_query(
        query,
        c,
        parse_dates=["data"],
        params=params if params else None
    )

def faturamento_por_descricao(
    data_inicio: Optional[str] = None, 
    data_fim: Optional[str] = None
) -> pd.DataFrame:
    c = _conn()
    
    query = """
        SELECT descricao, SUM(valor) AS total
        FROM transacoes
        WHERE tipo='faturamento'
    """
    
    conditions: list[str] = []
    params: list[str] = []
    
    if data_inicio:
        conditions.append("date(data) >= ?")
        params.append(data_inicio)
    if data_fim:
        conditions.append("date(data) <= ?")
        params.append(data_fim)
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += " GROUP BY descricao ORDER BY total DESC"
    return pd.read_sql_query(query, c, params=params if params else None)

def despesa_por_descricao(
    data_inicio: Optional[str] = None,
    data_fim: Optional[str] = None
) -> pd.DataFrame:
    """
    Agrupa e soma todas as transações do tipo 'despesa' por descrição,
    filtrando opcionalmente por período.
    """
    c = _conn()
    
    query = """
        SELECT descricao, SUM(valor) AS total
        FROM transacoes
        WHERE tipo='despesa'
    """
    
    conditions: list[str] = []
    params: list[str] = []
    
    if data_inicio:
        conditions.append("date(data) >= ?")
        params.append(data_inicio)
    if data_fim:
        conditions.append("date(data) <= ?")
        params.append(data_fim)
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += " GROUP BY descricao ORDER BY total DESC"
    return pd.read_sql_query(query, c, params=params if params else None)

# Padrão regex melhorado para capturar mais variações
PADRAO = re.compile(
    r"(?:registre|adicionar|inserir|incluir)\s+(?:r\$\s*)?([\d\.,]+)\s+de\s+"
    r"(faturamento|despesa|receita|gasto)s?\s+"
    r"(?:em|para|no dia|dia)\s+(\d{1,2}[\/-]\d{1,2}[\/-]\d{4})"
    r"(?:\s+(?:com|descrição|motivo)\s+['\"]?(.+?)['\"]?)?\s*$",
    flags=re.IGNORECASE
)

def tentar_extrair_comando(txt: str) -> Registro | None:
    m = PADRAO.search(txt.strip())
    if not m:
        return None
    valor_str = m.group(1).replace('.', '').replace(',', '.')
    try:
        valor = float(valor_str)
    except ValueError:
        return None

    # Normalizar a data: aceita tanto / quanto -
    data_str = m.group(3).replace('-', '/')
    try:
        data_iso = datetime.strptime(data_str, "%d/%m/%Y").date().isoformat()
    except ValueError:
        return None

    tipo = m.group(2).lower()
    # Mapear sinônimos para os tipos
    if tipo in ["receita"]:
        tipo = "faturamento"
    elif tipo in ["gasto"]:
        tipo = "despesa"

    descricao = m.group(4).strip() if m.group(4) else None

    return {
        "valor": valor,
        "tipo": tipo,
        "data": data_iso,
        "descricao": descricao,
    }

def buscar_transacao(
    valor: float | None = None,
    descricao: str | None = None,
    tipo: str | None = None,
    data: str | None = None,
    margem: float = 0.01
) -> list[dict[str, Any]]:
    """
    Busca transações com base em critérios fornecidos.
    Args:
        valor: Valor aproximado (busca com margem de erro)
        descricao: Parte da descrição (busca por substring)
        tipo: 'faturamento' ou 'despesa'
        data: Data exata (no formato 'YYYY-MM-DD')
        margem: Margem de erro para busca por valor (padrão 1%)
    
    Retorna uma lista de dicionários representando as transações.
    """
    c = _conn()
    query = "SELECT * FROM transacoes WHERE 1=1"
    params = []
    
    if valor is not None:
        # Considera uma margem relativa para evitar problemas com ponto flutuante
        margem_valor = valor * margem
        query += " AND valor BETWEEN ? AND ?"
        params.extend([valor - margem_valor, valor + margem_valor])
    
    if descricao:
        query += " AND descricao LIKE ?"
        params.append(f"%{descricao}%")
    
    if tipo:
        query += " AND tipo = ?"
        params.append(tipo)
    
    if data:
        query += " AND date(data) = ?"
        params.append(data)
    
    query += " ORDER BY data DESC"
    
    cursor = c.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]

def buscar_por_valor(valor: float, margem: float = 0.01) -> list[dict]:
    """Função de compatibilidade para buscas rápidas por valor"""
    return buscar_transacao(valor=valor, margem=margem)