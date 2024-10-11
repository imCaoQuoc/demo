import os
import ollama
import streamlit as st
import nest_asyncio
from streamlit_chat import message
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

nest_asyncio.apply()

documents = SimpleDirectoryReader(
    input_files=["document/AI.txt"]
).load_data()

combined_documents = Document(text="\n\n".join([doc.text for doc in documents]))

def generate_response(prompt):
    response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def get_build_index(documents, embed_model="BAAI/bge-small-en-v1.5", save_dir="./vector_store/index"):
    # Set index settings
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Check if the save directory exists
    if not os.path.exists(save_dir):
        # Create and load the index
        index = VectorStoreIndex.from_documents(
            [combined_documents], service_context=Settings
        )
        index.storage_context.persist(persist_dir=save_dir)
    else:
        # Load the existing index
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=Settings,
        )
    return index

def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return engine

# Get the Vector Index
vector_index = get_build_index(documents=documents, embed_model="BAAI/bge-small-en-v1.5", save_dir="./vector_store/index")

# Create a query engine with the specified parameters
query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=8, rerank_top_n=5)

# Query the engine with a question
query = st.chat_input()
if query:
    message(query, is_user=True)
    response_1 = query_engine.query(query)
    prompt = f'''Generate a detailed response for the query asked based only on the context fetched:
                Query: {query}
                Context: {response_1}

                Instructions:
                1. Show query and your generated response based on context.
                2. Your response should be detailed and should cover every aspect of the context.
                3. Be crisp and concise.
                4. Don't include anything else in your response - no header/footer/code etc
                '''
    response_2 = generate_response(prompt)
    message(response_2)