import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Estilos personalizados para la UI (botón gris)
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #808080;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("Buscador de Papers Científicos (RAG)")

@st.cache_resource
def load_rag_system():
    # 1. Cargar embeddings y Vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./arxiv_chroma_db_full", embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 2. Cargar LLM (usando el modelo actualizado y la clave segura)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=os.environ.get("GROQ_API_KEY"))

    # 3. Prompt estricto
    template = """
    You are a helpful assistant for answering questions based on academic paper abstracts.
    Use the following pieces of retrieved context to answer the question. 
    If the information to answer the question is not present in the context, explicitly state: "El corpus no contiene información suficiente para responder a esta consulta."
    Do not make up information.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # 4. Funciones y Cadena RAG con LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    setup_and_retrieval = RunnableParallel(
        {
            "source_documents": (lambda x: x["query"]) | retriever, 
            "question": lambda x: x["query"]
        }
    )

    qa_chain = (
        setup_and_retrieval
        | RunnablePassthrough.assign(context=(lambda x: format_docs(x["source_documents"])))
        | RunnablePassthrough.assign(result=(prompt | llm | StrOutputParser()))
    )

    return qa_chain

try:
    qa_chain = load_rag_system()
    st.success("✅ Base de datos y sistema RAG listos.")
except Exception as e:
    st.error("Error cargando el sistema. Verifica la carpeta de ChromaDB.")

query = st.text_input("Ingresa tu consulta sobre papers:")

if st.button("Consultar"):
    if query:
        with st.spinner("Generando respuesta..."):
            # Invocamos la cadena LCEL
            resultado = qa_chain.invoke({"query": query})

            st.write("### Respuesta")
            st.write(resultado['result'])

            st.write("### Evidencias")
            # Extraemos los documentos devueltos por RunnableParallel
            for i, doc in enumerate(resultado['source_documents']):
                with st.expander(f"Evidencia {i+1}: {doc.metadata.get('title', 'Documento')}"):
                    st.write(doc.page_content)
    else:
        st.warning("Por favor ingresa una consulta válida.")
