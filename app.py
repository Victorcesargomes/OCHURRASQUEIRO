from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

########################
# Configurações gerais #
########################
BASE_PATH = Path(__file__).parent
CSV_PATH = BASE_PATH / "dados.csv"
CERT_PATH = BASE_PATH / "certidoes"
LOGO_PATH = BASE_PATH / "logoo.png"
CLIENT_NAME = "O Churrasqueiro"
MODEL_NAME = "llama-3.1-8b-instant"

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
CONTABIL_API_URL = os.getenv("CONTABIL_API_URL")

#########################
# Configuração do logger #
#########################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.FileHandler(BASE_PATH / "app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

######################################
# Index simples das certidões locais #
######################################

def indexar_certidoes(pasta: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if pasta.exists():
        for arquivo in pasta.glob("*.pdf"):
            slug = (
                arquivo.stem.lower()
                .replace("_", " ")
                .replace("-", " ")
                .replace("  ", " ")
                .strip()
            )
            slug_ascii = (
                unicodedata.normalize("NFKD", slug)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
            index[slug_ascii] = arquivo
    return index


CERTIDOES = indexar_certidoes(CERT_PATH)
logger.info("Certidões indexadas: %s", list(CERTIDOES.keys()))

###############################
# Utilitários de normalização #
###############################

def normalizar_txt(txt: str) -> str:
    ascii_txt = (
        unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    )
    ascii_txt = re.sub(r"[^a-z0-9 ]", " ", ascii_txt.lower())
    ascii_txt = re.sub(r"\s+", " ", ascii_txt).strip()
    return ascii_txt

###############################
# Funções utilitárias do CSV  #
###############################

@st.cache_data(show_spinner=False)
def carregar_df(caminho: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho, sep=";", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        def _parse_money(txt: str) -> float:
            if pd.isna(txt) or str(txt).strip() == "":
                return 0.0
            txt = str(txt).strip().replace("\u00A0", "")
            txt = txt.replace(".", "").replace(",", ".")
            try:
                return float(txt)
            except ValueError:
                logger.warning("Valor monetário inválido ignorado: %r", txt)
                return 0.0

        for col in ("faturamento", "despesa", "lucro"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_money)
            else:
                logger.warning("Coluna '%s' ausente no CSV.", col)
                df[col] = 0.0

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

        datas_invalidas = df["data"].isna().sum()
        if datas_invalidas:
            logger.warning("%d linha(s) com datas inválidas foram ignoradas.", datas_invalidas)

        return df
    except FileNotFoundError:
        logger.error("Arquivo CSV não encontrado: %s", caminho)
        return pd.DataFrame()
    except Exception as exc:
        logger.exception("Erro ao carregar CSV: %s", exc)
        return pd.DataFrame()


def df_para_prompt(df: pd.DataFrame) -> str:
    if df.empty:
        return "Nenhum dado disponível."
    df_display = df.copy()
    for col in ("faturamento", "despesa", "lucro"):
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
    MAX_LINHAS_PROMPT = 200
    if len(df_display) > MAX_LINHAS_PROMPT:
        logger.info("DataFrame truncado para %d linhas no prompt.", MAX_LINHAS_PROMPT)
        df_display = df_display.tail(MAX_LINHAS_PROMPT)
    return df_display.to_csv(index=False, sep=";")


dados_df = carregar_df(CSV_PATH)

faturamento_total = dados_df["faturamento"].sum() if not dados_df.empty and "faturamento" in dados_df.columns else 0.0
despesa_total     = dados_df["despesa"].sum()     if not dados_df.empty and "despesa" in dados_df.columns else 0.0
lucro_total       = dados_df["lucro"].sum()       if not dados_df.empty and "lucro" in dados_df.columns else 0.0

##############################
# Análise Financeira Avançada #
##############################

def analisar_financas(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "top_despesas": pd.DataFrame(),
            "faturamento_medio_diario": 0.0,
            "margem_lucro": 0.0,
            "despesas_recorrentes": pd.Series(),
        }

    top_despesas = df.nlargest(5, "despesa")[["descricao", "despesa"]] if "descricao" in df.columns else pd.DataFrame()
    faturamento_medio_diario = df[df["faturamento"] > 0]["faturamento"].mean() if not df.empty else 0.0
    margem_lucro = (df["lucro"].sum() / df["faturamento"].sum() * 100) if df["faturamento"].sum() > 0 else 0.0
    despesas_recorrentes = (
        df[df["despesa"] > 0].groupby("descricao")["despesa"].sum().nlargest(5)
        if "descricao" in df.columns else pd.Series()
    )

    return {
        "top_despesas": top_despesas,
        "faturamento_medio_diario": faturamento_medio_diario,
        "margem_lucro": margem_lucro,
        "despesas_recorrentes": despesas_recorrentes,
    }


analise = analisar_financas(dados_df)

############################
# Visualizações de Dados   #
############################

def plot_despesas(df: pd.DataFrame):
    if df.empty or "descricao" not in df.columns:
        return None
    despesas_agrupadas = (
        df[df["despesa"] > 0].groupby("descricao")["despesa"].sum().nlargest(10)
    )
    if despesas_agrupadas.empty:
        return None
    despesas_df = despesas_agrupadas.reset_index()
    despesas_df.columns = ["descricao", "valor_total"]
    fig = px.pie(
        despesas_df, names="descricao", values="valor_total",
        title="Top 10 Despesas (Valor Total)", hole=0.3,
    )
    fig.update_traces(
        textposition="inside", textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>R$ %{value:,.2f} (%{percent})",
    )
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode="hide",
                      margin=dict(t=50, b=20, l=20, r=20))
    return fig


def plot_evolucao(df: pd.DataFrame, data_inicio: datetime, data_fim: datetime):
    if df.empty or "data" not in df.columns:
        return None
    df_periodo = df[
        (df["data"] >= pd.Timestamp(data_inicio))
        & (df["data"] <= pd.Timestamp(data_fim))
    ]
    if df_periodo.empty:
        return None
    df_agrupado = df_periodo.groupby("data")[["faturamento", "despesa", "lucro"]].sum().reset_index()
    fig = px.line(
        df_agrupado,
        x="data",
        y=["faturamento", "despesa", "lucro"],
        title=f"Evolução Financeira: {data_inicio.strftime('%d/%m/%Y')} - {data_fim.strftime('%d/%m/%Y')}",
        labels={"value": "Valor (R$)", "variable": "Indicador"},
        color_discrete_map={"faturamento": "#2ecc71", "despesa": "#e74c3c", "lucro": "#3498db"},
    )
    fig.update_layout(hovermode="x unified", legend_title_text="",
                      xaxis_title="Data", yaxis_title="Valor (R$)",
                      margin=dict(t=50, b=50, l=50, r=50))
    fig.update_traces(hovertemplate="<b>%{y:,.2f}</b>", mode="lines+markers")
    return fig

###################################
# Histórico de mensagens          #
###################################

def get_historico() -> List[BaseMessage]:
    return st.session_state.setdefault("historico", [])


def adicionar_mensagem(role: str, conteudo: str) -> None:
    historico = get_historico()
    msg = HumanMessage(content=conteudo) if role == "human" else AIMessage(content=conteudo)
    historico.append(msg)
    if len(historico) > 40:
        st.session_state["historico"] = historico[-40:]


def limpar_historico() -> None:
    st.session_state["historico"] = []

###################################
# Configuração do modelo (Groq)   #
###################################

@st.cache_resource
def _criar_client() -> ChatGroq:
    return ChatGroq(api_key=API_KEY, model=MODEL_NAME)


def _criar_chain(llm: ChatGroq):
    system_message = f"""
Você é Victor, assistente virtual da hamburgueria \"{CLIENT_NAME}\" do Ibson.
Fale **sempre** em português brasileiro, de forma clara e objetiva.
Ignore qualquer texto entre as tags <think> e </think>; trate-o como nota interna.
Você interage exclusivamente com o Ibson, dono da hamburgueria.

**DIRETRIZES DE RESPOSTA:**
1. Seja conciso. Evite textos longos quando a pergunta for simples.
2. Para perguntas objetivas, responda em no máximo 2 frases.
3. Para perguntas sobre valores financeiros, mostre apenas os números relevantes.
4. Para pedidos de certidões, forneça imediatamente o link de download.
5. Use negrito apenas para valores numéricos importantes.

**Resumo financeiro até {datetime.now().strftime('%d/%m/%Y')}**:
- **Faturamento acumulado:** R$ {faturamento_total:,.2f}
- **Despesa acumulada:** R$ {despesa_total:,.2f}
- **Lucro acumulado:** R$ {lucro_total:,.2f}

**Análise detalhada**:
- **Top 5 despesas**:
{analise['top_despesas'].to_string(index=False) if not analise['top_despesas'].empty else 'Nenhuma despesa registrada'}
- **Faturamento médio diário**: R$ {analise['faturamento_medio_diario']:,.2f}
- **Margem de lucro**: {analise['margem_lucro']:.2f}%
- **Principais despesas recorrentes**:
{analise['despesas_recorrentes'].to_string() if not analise['despesas_recorrentes'].empty else 'Nenhuma despesa recorrente'}

Certidões disponíveis: {', '.join(CERTIDOES.keys()) or 'nenhuma'}

Base de dados detalhada:
###
{df_para_prompt(dados_df)}
###

- As colunas correspondem a `Data`, `Faturamento`, `Despesa`, `Descrição` e `Lucro`.
- Todos os valores estão em Reais (BRL).

Se precisar de uma certidão, basta pedir — exemplos: "quero a CND estadual", "quero a CND FGTS".

Termine perguntando se precisa de algo mais.
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ])
    return prompt_template | llm

#######################################
# Certidões PDF                        #
#######################################

CATEGORIAS_CERT = {
    "estadual": "Estadual", "federal": "Federal",
    "municipal": "Municipal", "fgts": "FGTS", "fiscal": "Fiscal",
}


def tentar_enviar_certidao(mensagem: str) -> Tuple[Optional[Path], Optional[str]]:
    txt = normalizar_txt(mensagem)
    if "cnd" not in txt and "certidao" not in txt:
        return None, None
    for cat_slug, cat_nome in CATEGORIAS_CERT.items():
        if cat_slug in txt:
            for slug, path in CERTIDOES.items():
                if cat_slug in slug and ("cnd" in slug or "certidao" in slug):
                    return path, cat_nome
    return None, None

################################
# Contabilidade                 #
################################

def enviar_contabilidade(df: pd.DataFrame) -> bool:
    if not CONTABIL_API_URL:
        logger.warning("URL da API contábil não configurada.")
        return False
    try:
        dados_json = df.assign(
            data=df["data"].dt.strftime("%Y-%m-%d").fillna("")
        ).to_dict(orient="records")
        response = requests.post(CONTABIL_API_URL, json=dados_json, timeout=10)
        if response.status_code == 200:
            logger.info("Dados enviados para contabilidade com sucesso.")
            return True
        logger.error("Erro ao enviar: status=%s body=%s", response.status_code, response.text)
        return False
    except requests.exceptions.Timeout:
        logger.error("Timeout ao enviar dados para contabilidade.")
        return False
    except Exception as exc:
        logger.exception("Erro na integração contábil: %s", exc)
        return False

###########################
# LLM                      #
###########################

def consultar_modelo(chain, entrada: str) -> str:
    try:
        historico = get_historico()
        resposta = chain.invoke({
            "input": entrada,
            "chat_history": historico,
        }).content
        return resposta
    except Exception as exc:
        logger.exception("Erro na chamada do LLM: %s", exc)
        return "❌ Ocorreu um erro. Tente novamente."

###########################
# Interface Streamlit      #
###########################

def desenhar_sidebar(chain) -> None:
    with st.sidebar:
        st.image(LOGO_PATH, use_container_width=True)
        abas = st.tabs(["Conversas", "Configurações"])

        with abas[0]:
            if st.button("🗑️ Apagar Histórico", use_container_width=True):
                limpar_historico()
                st.success("Histórico apagado!")

        with abas[1]:
            st.header("⚙️ Configurações")
            st.markdown(
                f"""
                - **Modelo:** {MODEL_NAME}
                - **Usuário autorizado:** Ibson ({CLIENT_NAME})
                - **Linhas CSV:** {len(dados_df)}
                - **Certidões disponíveis:** {', '.join(CERTIDOES) or 'nenhuma'}
                """
            )
            if CONTABIL_API_URL:
                if st.button("📤 Enviar para Contabilidade"):
                    if enviar_contabilidade(dados_df):
                        st.success("Enviado com sucesso!")
                    else:
                        st.error("Falha no envio. Verifique os logs.")
            else:
                st.info("Integração contábil não configurada.")


def pagina_chat(chain) -> None:
    st.header(f"📊 Dashboard Financeiro - {CLIENT_NAME}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Faturamento Total", f"R$ {faturamento_total:,.2f}")
    col2.metric("Despesa Total",     f"R$ {despesa_total:,.2f}")
    col3.metric("Lucro Total",       f"R$ {lucro_total:,.2f}")

    if not dados_df.empty:
        st.subheader("Top 10 Despesas")
        fig = plot_despesas(dados_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma despesa registrada.")

    st.divider()
    if not dados_df.empty and "data" in dados_df.columns:
        st.subheader("📈 Análise Temporal")
        datas_validas = dados_df["data"].dropna()
        if datas_validas.empty:
            st.warning("Nenhuma data válida no CSV.")
        else:
            min_date = datas_validas.min().to_pydatetime().date()
            max_date = datas_validas.max().to_pydatetime().date()
            c1, c2 = st.columns(2)
            data_inicio = c1.date_input("Data inicial", min_date, min_value=min_date, max_value=max_date)
            data_fim    = c2.date_input("Data final",   max_date, min_value=min_date, max_value=max_date)

            if data_inicio > data_fim:
                st.warning("A data inicial não pode ser posterior à data final.")
            else:
                fig_evol = plot_evolucao(
                    dados_df,
                    datetime.combine(data_inicio, datetime.min.time()),
                    datetime.combine(data_fim,    datetime.max.time()),
                )
                if fig_evol:
                    st.plotly_chart(fig_evol, use_container_width=True)
                else:
                    st.info("Sem dados no período selecionado.")

    st.divider()
    st.subheader("💬 Conversa com o Analista")

    # Exibir histórico
    for msg in get_historico():
        if isinstance(msg, HumanMessage):
            st.chat_message("human").markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("ai"):
                st.markdown(msg.content)
                col_fb1, col_fb2, _ = st.columns([0.1, 0.1, 0.8])
                with col_fb1:
                    if st.button("👍", key=f"up_{hash(msg.content)}"):
                        st.toast("Obrigado pelo feedback positivo!")
                with col_fb2:
                    if st.button("👎", key=f"dw_{hash(msg.content)}"):
                        st.toast("Obrigado pelo feedback. Vou melhorar!")

    entrada = st.chat_input("Fale com o Analista")
    if not entrada:
        return

    entrada_limpa = re.sub(
        r"<think>.*?</think>", "", entrada, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if not entrada_limpa:
        st.info("(Comentário interno ignorado.)")
        return

    st.chat_message("human").markdown(entrada_limpa)
    adicionar_mensagem("human", entrada_limpa)

    cert_path, categoria = tentar_enviar_certidao(entrada_limpa)
    if cert_path and categoria:
        resposta = f"Aqui está a CND {categoria} conforme solicitado."
        st.chat_message("ai").markdown(resposta)
        with open(cert_path, "rb") as f:
            st.download_button(f"📄 Baixar CND {categoria}", data=f.read(),
                               file_name=cert_path.name, mime="application/pdf")
        adicionar_mensagem("ai", resposta + f" [CND {categoria} anexada]")
        return

    with st.chat_message("ai"):
        resposta_llm = consultar_modelo(chain, entrada_limpa)
        st.markdown(resposta_llm)
        col_fb1, col_fb2, _ = st.columns([0.1, 0.1, 0.8])
        with col_fb1:
            if st.button("👍", key=f"up_{hash(resposta_llm)}"):
                st.toast("Obrigado pelo feedback positivo!")
        with col_fb2:
            if st.button("👎", key=f"dw_{hash(resposta_llm)}"):
                st.toast("Obrigado pelo feedback. Vou melhorar!")

    adicionar_mensagem("ai", resposta_llm)


def main() -> None:
    if not API_KEY:
        st.error("Variável GROQ_API_KEY não encontrada. Configure o arquivo .env.")
        st.stop()

    llm = _criar_client()

    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = _criar_chain(llm)

    desenhar_sidebar(st.session_state["llm_chain"])
    pagina_chat(st.session_state["llm_chain"])


if __name__ == "__main__":
    main()
