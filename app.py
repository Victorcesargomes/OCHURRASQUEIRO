import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")

# Memória da conversa
MEMORIA = ConversationBufferWindowMemory(k=5, return_messages=True)

# Carregar CSV
CAMINHO_CSV = "dados.csv"

def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

dados_documento = carrega_csv(CAMINHO_CSV)

# System Message personalizado
system_message = f"""
Você é Victor, assistente virtual da hamburgueria "O Churrasqueiro".
Você interage exclusivamente com Ibson Dantas.

Dados disponíveis:
###
{dados_documento}
###

- "valor_das": DAS mensal.
- "data_pagamento": data do pagamento do DAS.
- Troque "$" por "S".

Se precisar de mais dados, solicite educadamente.

Termine perguntando se precisa algo mais.
"""

template = ChatPromptTemplate.from_messages([
    ('system', system_message),
    ('placeholder', '{chat_history}'),
    ('user', '{input}')
])

chain = template | client

# Interface do chat
def pagina_chat():
    st.header('🤖 Analista Contábil')

    memoria = st.session_state.get('memoria', MEMORIA)

    for mensagem in memoria.buffer_as_messages:
        chat_display = st.chat_message(mensagem.type)
        chat_display.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o Analista')

    if input_usuario:
        chat_display = st.chat_message('human')
        chat_display.markdown(input_usuario)

        chat_display = st.chat_message('ai')
        resposta = chat_display.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
        }))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)

        st.session_state['memoria'] = memoria

# Sidebar fixada corretamente
def sidebar():
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        tabs = st.tabs(['Conversas', 'Configurações'])

        with tabs[0]:
            if st.button('🗑️ Apagar Histórico'):
                st.session_state['memoria'] = MEMORIA
                st.success("Histórico apagado com sucesso!")

        with tabs[1]:
            st.header("⚙️ Configurações")
            st.markdown("""
            - **Modelo:** Llama 3.3-70b
            - **Usuário autorizado:** Ibson Dantas
            - **Empresa:** O Churrasqueiro 🍔
            """)

# Execução principal
def main():
    sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
