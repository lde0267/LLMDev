from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import tiktoken

file_path = "./data/hood2.pdf"

tokenizer = tiktoken.encoding_for_model("gpt-4o")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


loader = PyPDFLoader(file_path)
# 불러오고 토큰 단위로 자르기기
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function = tiktoken_len
)

# document 객체로 만들기
docs = text_splitter.split_documents(pages)

# 임베딩
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db2 = Chroma.from_documents(docs, hf, persist_directory= "./data/chroma_db")