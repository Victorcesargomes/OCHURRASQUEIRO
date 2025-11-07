from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from dotenv import load_dotenv
from langchain_community.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

########################
# Configurações gerais #
########################
BASE_PATH = Path(__file__).parent
CSV_PATH = BASE_PATH / "dados.csv"
CERT_PATH = BASE_PATH / "certidoes"
LOGO_PATH = BASE_PATH / "logoo.png"
CLIENT_NAME = "O Churrasqueiro"
MODEL_NAME = "llama-3.1-8b-instant"  # Modelo Groq válido

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
CONTABIL_API_URL = os.getenv("CONTABIL_API_URL")

if not API_KEY:
    st.error("Variável GROQ_API_KEY não encontrada.")
    st.stop()

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
    """Cria um dicionário {slug_ascii: filepath} para pesquisa rápida."""
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
            slug_ascii = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode("ascii")
            index[slug_ascii] = arquivo
    return index

CERTIDOES = indexar_certidoes(CERT_PATH)
logger.info("Certidões indexadas: %s", CERTIDOES)

###############################
# Utilitários de normalização #
###############################

def normalizar_txt(txt: str) -> str:
    """Remove acentos, caixa alta e símbolos não alfanuméricos."""
    ascii_txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    ascii_txt = re.sub(r"[^a-z0-9 ]", " ", ascii_txt.lower())
    ascii_txt = re.sub(r"\s+", " ", ascii_txt).strip()
    return ascii_txt

###############################
# Funções utilitárias do CSV  #
###############################
@st.cache_data(show_spinner=False)
def carregar_df(caminho: Path) -> pd.DataFrame:
    """Lê o CSV do cliente com colunas: Data, faturamento, despesa, descricao, lucro."""
    try:
        df = pd.read_csv(caminho, sep=";", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        def _parse_money(txt: str):
            if pd.isna(txt) or txt == "":
                return 0.0
            txt = txt.strip().replace("\u00A0", "")
            txt = txt.replace(".", "").replace(",", ".")
            return float(txt)

        for col in ("faturamento", "despesa", "lucro"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_money)

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
        return df
    except Exception as exc:
        logger.exception("Erro ao carregar CSV: %s", exc)
        st.error(f"Erro ao carregar {caminho.name}: {exc}")
        return pd.DataFrame()

def df_para_prompt(df: pd.DataFrame) -> str:
    """Formata o DataFrame para exibição no prompt, com valores em Reais formatados."""
    if df.empty:
        return "Nenhum dado disponível."
    
    df_display = df.copy()
    # Formata as colunas numéricas para o padrão brasileiro
    for col in ('faturamento', 'despesa', 'lucro'):
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f'R$ {x:,.2f}'.replace(",", "X").replace(".", ",").replace("X", ".")
            )
    return df_display.to_csv(index=False, sep=";")

dados_df = carregar_df(CSV_PATH)

# Resumos financeiros
faturamento_total = dados_df["faturamento"].sum() if not dados_df.empty and "faturamento" in dados_df.columns else 0.0
despesa_total = dados_df["despesa"].sum() if not dados_df.empty and "despesa" in dados_df.columns else 0.0
lucro_total = dados_df["lucro"].sum() if not dados_df.empty and "lucro" in dados_df.columns else 0.0

##############################
# Análise Financeira Avançada #
##############################

def analisar_financas(df: pd.DataFrame) -> Dict[str, Any]:
    """Realiza análise detalhada das finanças."""
    if df.empty:
        return {}
    
    analise = {}
    
    # Top 5 despesas
    if 'despesa' in df.columns and 'descricao' in df.columns:
        top_despesas = df.nlargest(5, 'despesa')[['descricao', 'despesa']] if not df.empty else pd.DataFrame()
        analise["top_despesas"] = top_despesas
    else:
        analise["top_despesas"] = pd.DataFrame()
    
    # Faturamento médio diário (considerando dias com faturamento)
    if 'faturamento' in df.columns:
        faturamento_medio_diario = df[df['faturamento'] > 0]['faturamento'].mean() if not df.empty else 0.0
        analise["faturamento_medio_diario"] = faturamento_medio_diario
    else:
        analise["faturamento_medio_diario"] = 0.0
        
    # Margem de lucro (lucro total / faturamento total)
    if (not df.empty and 'faturamento' in df.columns and 'lucro' in df.columns and 
        df['faturamento'].sum() > 0):
        margem_lucro = (df['lucro'].sum() / df['faturamento'].sum()) * 100
    else:
        margem_lucro = 0.0
    analise["margem_lucro"] = margem_lucro
    
    # Despesas recorrentes (agrupadas por descrição, somadas e as 5 maiores)
    if 'despesa' in df.columns and 'descricao' in df.columns:
        despesas_recorrentes = df[df['despesa'] > 0].groupby('descricao')['despesa'].sum().nlargest(5) if not df.empty else pd.Series()
        analise["despesas_recorrentes"] = despesas_recorrentes
    else:
        analise["despesas_recorrentes"] = pd.Series()
    
    return analise

analise = analisar_financas(dados_df)

############################
# Visualizações de Dados #
############################

def plot_despesas(df: pd.DataFrame):
    """Gráfico de pizza das top 10 despesas por valor total."""
    if df.empty or 'despesa' not in df.columns or 'descricao' not in df.columns:
        return None
        
    despesas_agrupadas = df[df['despesa'] > 0].groupby('descricao')['despesa'].sum().nlargest(10)
    if despesas_agrupadas.empty:
        return None
        
    despesas_df = despesas_agrupadas.reset_index()
    despesas_df.columns = ['descricao', 'valor_total']
    
    fig = px.pie(
        despesas_df,
        names='descricao',
        values='valor_total',
        title='Top 10 Despesas (Valor Total)',
        hole=0.3
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>R$ %{value:,.2f} (%{percent})'
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_evolucao(df: pd.DataFrame, data_inicio: datetime, data_fim: datetime):
    """Gráfico de linha da evolução financeira no período."""
    if df.empty or 'data' not in df.columns:
        return None
        
    df_periodo = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)]
    if df_periodo.empty:
        return None
        
    # Agrupar por dia e somar valores
    colunas_disponiveis = [col for col in ['faturamento', 'despesa', 'lucro'] if col in df_periodo.columns]
    if not colunas_disponiveis:
        return None
        
    df_agrupado = df_periodo.groupby('data')[colunas_disponiveis].sum().reset_index()
    
    fig = px.line(
        df_agrupado,
        x='data',
        y=colunas_disponiveis,
        title=f'Evolução Financeira: {data_inicio.strftime("%d/%m/%Y")} - {data_fim.strftime("%d/%m/%Y")}',
        labels={'value': 'Valor (R$)', 'variable': 'Indicador'},
        color_discrete_map={
            'faturamento': '#2ecc71',
            'despesa': '#e74c3c',
            'lucro': '#3498db'
        }
    )
    
    fig.update_layout(
        hovermode='x unified',
        legend_title_text='',
        xaxis_title='Data',
        yaxis_title='Valor (R$)',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y:,.2f}</b>',
        mode='lines+markers'
    )
    
    return fig

###################################
# Configuração do modelo (Groq LLM) #
###################################
try:
    client = ChatGroq(api_key=API_KEY, model=MODEL_NAME)
    MEMORIA_PADRAO = ConversationSummaryBufferMemory(
        llm=client,
        max_token_limit=2000,
        return_messages=True
    )
except Exception as e:
    logger.error(f"Erro ao inicializar cliente Groq: {e}")
    st.error(f"Erro ao inicializar o assistente: {e}")
    st.stop()

# Preparar a mensagem do sistema
system_message = f"""
Você é Victor, assistente virtual da hamburgueria \"{CLIENT_NAME}\" do Ibson.
Fale **sempre** em português brasileiro, de forma clara e objetiva.
Ignore qualquer texto entre as tags <think> e </think>; trate‑o como nota interna que NÃO deve ser respondida nem exibida.
Você interage exclusivamente com o Ibson, dono da hamburgueria.

**DIRETRIZES DE RESPOSTA:**
1. Seja conciso em respostas diretas. Evite textos longos quando a pergunta for simples.
2. Para perguntas objetivas, responda de forma direta com no máximo 2 frases.
3. Para perguntas sobre valores financeiros, mostre apenas os números relevantes.
4. Para pedidos de certidões, forneça imediatamente o link de download.
5. Respostas devem ocupar no máximo 3 linhas quando possível.
6. Use negrito apenas para valores numéricos importantes.

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

Eu também tenho acesso às seguintes certidões (PDF):
{', '.join(CERTIDOES.keys()) or 'nenhuma'}

Base de dados detalhada:
###
{df_para_prompt(dados_df)}
###

- As colunas correspondem a `Data`, `Faturamento`, `Despesa`, `Descrição` e `Lucro`.
- Todos os valores estão em Reais (BRL).

Se precisar de uma certidão, basta pedir — exemplos: “quero a CND estadual”, “quero a CND FGTS”, "quero a CND federal", "quero a CND fiscal etc.

Termine perguntando se precisa de algo mais.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
])

chain = prompt_template | client

#######################################
# Funções para envio de certidões PDF #
#######################################

CATEGORIAS_CERT = {
    "estadual": "Estadual",
    "federal": "Federal",
    "municipal": "Municipal",
    "fgts": "FGTS",
    "fiscal": "Fiscal",
}

def tentar_enviar_certidao(mensagem: str) -> Tuple[Optional[Path], Optional[str]]:
    """Retorna (Path, categoria) da certidão solicitada ou (None, None)."""
    txt = normalizar_txt(mensagem)

    # Verifica se mensagem contém 'cnd' ou 'certidao'
    if "cnd" not in txt and "certidao" not in txt:
        return None, None

    for cat_slug, cat_nome in CATEGORIAS_CERT.items():
        if cat_slug in txt:
            # Procura arquivo cujo slug contenha a categoria
            for slug, path in CERTIDOES.items():
                if cat_slug in slug and ("cnd" in slug or "certidao" in slug):
                    return path, cat_nome
    return None, None

################################
# Integração com Contabilidade #
################################

def enviar_contabilidade(df: pd.DataFrame) -> bool:
    """Envia dados para sistema contábil externo."""
    if not CONTABIL_API_URL:
        logger.warning("URL da API contábil não configurada.")
        return False
    try:
        # Converter DataFrame para JSON
        dados_json = df.to_dict(orient='records')
        response = requests.post(CONTABIL_API_URL, json=dados_json, timeout=10)
        if response.status_code == 200:
            logger.info("Dados enviados para contabilidade com sucesso.")
            return True
        logger.error("Erro ao enviar dados: %s", response.text)
        return False
    except Exception as exc:
        logger.exception("Erro na integração contábil: %s", exc)
        return False

###########################
# Funções para o modelo   #
###########################

@st.cache_data(show_spinner="Processando sua pergunta...")
def consultar_modelo(_memoria: ConversationSummaryBufferMemory, entrada: str) -> str:
    """Consulta o modelo com cache para melhor desempenho."""
    try:
        resposta = chain.invoke({
            "input": entrada,
            "chat_history": _memoria.chat_memory.messages
        }).content
        return resposta
    except Exception as exc:
        logger.exception("Erro na chamada do LLM: %s", exc)
        return "❌ Ocorreu um erro ao processar sua solicitação."

###########################
# Interface Streamlit     #
###########################

def desenhar_sidebar() -> None:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.warning("Logo não encontrada")
            
        abas = st.tabs(["Conversas", "Configurações"])

        with abas[0]:
            if st.button("🗑️ Apagar Histórico", use_container_width=True):
                st.session_state["memoria"] = ConversationSummaryBufferMemory(
                    llm=client,
                    max_token_limit=2000,
                    return_messages=True
                )
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
                if st.button("📤 Enviar Dados para Contabilidade"):
                    if enviar_contabilidade(dados_df):
                        st.success("Dados enviados com sucesso!")
                    else:
                        st.error("Falha no envio. Verifique logs.")
            else:
                st.info("Integração contábil não configurada")

def pagina_chat() -> None:
    st.header(f"📊 Dashboard Financeiro - {CLIENT_NAME}")
    
    # Métricas rápidas
    col1, col2, col3 = st.columns(3)
    col1.metric("Faturamento Total", f"R$ {faturamento_total:,.2f}")
    col2.metric("Despesa Total", f"R$ {despesa_total:,.2f}")
    col3.metric("Lucro Total", f"R$ {lucro_total:,.2f}")
    
    # Gráfico de despesas
    if not dados_df.empty:
        st.subheader("Top 10 Despesas (Valor Total)")
        fig = plot_despesas(dados_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma despesa registrada para exibir o gráfico.")
    
    # Análise temporal
    st.divider()
    if not dados_df.empty and "data" in dados_df.columns:
        st.subheader("📈 Análise Temporal")
        
        # Filtro por período
        min_date = dados_df["data"].min().to_pydatetime()
        max_date = dados_df["data"].max().to_pydatetime()
        
        col1, col2 = st.columns(2)
        with col1:
            data_inicio = st.date_input("Data inicial", min_date)
        with col2:
            data_fim = st.date_input("Data final", max_date)
        
        # Converter para datetime
        data_inicio_dt = datetime.combine(data_inicio, datetime.min.time())
        data_fim_dt = datetime.combine(data_fim, datetime.max.time())
        
        # Gráfico de evolução
        fig_evol = plot_evolucao(dados_df, data_inicio_dt, data_fim_dt)
        if fig_evol:
            st.plotly_chart(fig_evol, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para o período selecionado.")
    
    # Histórico de conversa
    st.divider()
    st.subheader("💬 Conversa com o Analista")
    memoria = st.session_state.get("memoria", MEMORIA_PADRAO)

    # Exibir histórico
    if hasattr(memoria, 'chat_memory') and hasattr(memoria.chat_memory, 'messages'):
        for i, msg in enumerate(memoria.chat_memory.messages):
            if isinstance(msg, HumanMessage):
                st.chat_message("human").markdown(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("ai").markdown(msg.content)
                # Adicionar botões de feedback para respostas do assistente
                col_fb1, col_fb2, _ = st.columns([0.1, 0.1, 0.8])
                with col_fb1:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        logger.info(f"Feedback positivo: {msg.content[:100]}")
                        st.toast("Obrigado pelo feedback positivo!")
                with col_fb2:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        logger.info(f"Feedback negativo: {msg.content[:100]}")
                        st.toast("Obrigado pelo feedback. Vou melhorar!")

    entrada = st.chat_input("Fale com o Analista")
    if not entrada:
        return

    # Remover blocos <think>…</think>
    entrada_limpa = re.sub(r"<think>.*?</think>", "", entrada, flags=re.DOTALL | re.IGNORECASE).strip()
    if not entrada_limpa:
        st.info("(Comentário interno ignorado.)")
        return

    st.chat_message("human").markdown(entrada_limpa)

    # Verificar pedido de certidão
    cert_path, categoria = tentar_enviar_certidao(entrada_limpa)
    if cert_path and categoria:
        resposta = f"Aqui está a CND {categoria} conforme solicitado."
        bot_msg = st.chat_message("ai")
        bot_msg.markdown(resposta)
        with open(cert_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            f"📄 Baixar CND {categoria}",
            data=pdf_bytes,
            file_name=cert_path.name,
            mime="application/pdf",
        )
        memoria.chat_memory.add_user_message(entrada_limpa)
        memoria.chat_memory.add_ai_message(resposta + f" [CND {categoria} anexada]")
        st.session_state["memoria"] = memoria
        return

    # Caso contrário, consultar o LLM
    bot_container = st.chat_message("ai")
    try:
        resposta_llm = consultar_modelo(memoria, entrada_limpa)
        bot_container.markdown(resposta_llm)
        
        # Adicionar botões de feedback
        col_fb1, col_fb2, _ = st.columns([0.1, 0.1, 0.8])
        with col_fb1:
            if st.button("👍", key=f"thumbs_up_new"):
                logger.info(f"Feedback positivo: {resposta_llm[:100]}")
                st.toast("Obrigado pelo feedback positivo!")
        with col_fb2:
            if st.button("👎", key=f"thumbs_down_new"):
                logger.info(f"Feedback negativo: {resposta_llm[:100]}")
                st.toast("Obrigado pelo feedback. Vou melhorar!")
                
    except Exception as exc:
        logger.exception("Erro na chamada do LLM: %s", exc)
        resposta_llm = "❌ Ocorreu um erro ao processar sua solicitação. Por favor, tente novamente."
        bot_container.error(resposta_llm)

    memoria.chat_memory.add_user_message(entrada_limpa)
    memoria.chat_memory.add_ai_message(resposta_llm)
    st.session_state["memoria"] = memoria

def main() -> None:
    # Inicializar memória se não existir
    if "memoria" not in st.session_state:
        st.session_state["memoria"] = MEMORIA_PADRAO
    
    desenhar_sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
