import wikipedia
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

nest_asyncio.apply()

Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Lấy dữ liệu từ Wikipedia
topics = ["Artificial Intelligence"]
for topic in topics:
    page = wikipedia.page(topic)
    
    with open('document/AI.txt', 'w', encoding='utf-8') as f: #Lưu file vào trong local
        f.write(page.content)