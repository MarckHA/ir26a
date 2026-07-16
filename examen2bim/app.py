import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gdown
import zipfile

# Configuración de página
st.set_page_config(page_title="Buscador Arxiv RAG", layout="centered")

st.title("🔎 Buscador de Papers Científicos (RAG)")

@st.cache_resource
def load_rag_system():
    persist_dir = "./arxiv_chroma_db_full"
    zip_filename = 'base.zip'
    
    # Si la carpeta no existe, la descargamos
    if not os.path.exists(persist_dir):
        st.info("Descargando base de datos...")
        file_id = os.environ.get("DRIVE_FILE_ID")
        gdown.download(id=file_id, output='base.zip', quiet=False)
        
        # Descomprimir en la raíz
        with zipfile.ZipFile('base.zip', 'r') as zip_ref:
            zip_ref.extractall(".")
            
    # VALIDACIÓN: Si después de descargar, la carpeta sigue sin existir, 
    # busca si se creó con otro nombre o en otra ruta
    if not os.path.exists(persist_dir):
        st.error(f"Error: La carpeta {persist_dir} no se creó tras la descarga. Verifica la estructura del ZIP.")
        st.stop()

    # 2. Cargar embeddings y base de datos con nombre de colección explícito
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # IMPORTANTE: Aquí especificamos 'arxiv_collection' para que coincida con tu notebook
    vector_db = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding_model,
        collection_name="arxiv_collection" 
    )

    # Debug: ¿Cuántos documentos ve Chroma realmente?
    try:
        doc_count = vector_db._collection.count()
        st.write(f"DEBUG: Documentos encontrados en la base: {doc_count}")
    except Exception as e:
        st.write(f"DEBUG: Error contando documentos: {e}")
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=os.environ.get("GROQ_API_KEY"))

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

# Ejecución principal
try:
    qa_chain = load_rag_system()
    st.success("✅ Sistema RAG conectado correctamente.")
except Exception as e:
    st.error(f"Error crítico: {e}")
    st.info("Nota: La base de datos vectorial no está presente en el repositorio debido a su tamaño. Por favor, asegúrate de que la carpeta 'arxiv_chroma_db_full' esté en el directorio raíz.")

query = st.text_input("Ingresa tu consulta sobre papers:")

if st.button("Consultar"):
    if query:
        with st.spinner("Buscando y generando respuesta..."):
            try:
                resultado = qa_chain.invoke({"query": query})
                st.write("### Respuesta")
                st.write(resultado['result'])

                st.write("### Evidencias consultadas")
                for i, doc in enumerate(resultado['source_documents']):
                    with st.expander(f"Evidencia {i+1}"):
                        st.write(doc.page_content)
            except Exception as e:
                st.error("Error al procesar la consulta.")
    else:
        st.warning("Por favor ingresa una consulta.")
