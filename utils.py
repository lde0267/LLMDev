import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA

import os
os.environ["OPENAI_API_KEY"] = "2sg8kxsseRytW3HOGXaGe1ESnMlAz9qGW1vpZ6EpkmQbCP2FfHdJJQQJ99BAACfhMk5XJ3w3AAAAACOGmcz2"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://youngwook-ai.openai.azure.com"

# PDF 파일을 로드하고 텍스트 분할
# def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=50):
#     loader = PyPDFLoader(file_path)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     texts = text_splitter.split_documents(pages)
#     return texts

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    # 각 페이지를 그대로 반환
    texts = [page.extract_text() for page in pages]
    
    return texts

# 임베딩 및 데이터베이스 생성
def create_vector_db(texts, model_name="jhgan/ko-sbert-nli", persist_directory="./data/chroma_db"):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = Chroma.from_documents(texts, hf, persist_directory=persist_directory)
    db.persist()
    return db

# 로컬 벡터 데이터베이스 불러오기
def load_vector_db(persist_directory="./data/chroma_db", model_name="jhgan/ko-sbert-nli"):
    hf = HuggingFaceBgeEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=hf)
    return db

# Azure ChatGPT와 QA 체인 구성
def create_retrieval_qa(retriever, deployment_name="dev-gpt-4o-mini", temperature=0.3):
    opneai = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=temperature
    )
    qa = RetrievalQA.from_chain_type(
        llm=opneai,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa
