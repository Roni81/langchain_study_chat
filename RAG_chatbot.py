from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import TextLoader

documents = TextLoader('/Users/sungminhong/Documents/langchain_tut/chatbot/AI.txt').load()

from langchain.text_splitter import RecursiveCharactorSplitter

def split_docs(documents, chunk_size = 1000, chunk_overlap = 30):
    text_splitter = RecursiveCharactorSplitter(chunk_size = chunk_size,
                                               chunk_overlap = chunk_overlap)
    return text_splitter.split(documents)

docs = split_docs(documents)

print(docs[0])


