from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader

doc_url = '/Users/sungminhong/Documents/langchain_tut/chatbot/AI.txt'

loader = TextLoader(doc_url)
documents = loader.load()


from langchain.text_splitter import CharacterTextSplitter


splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=15)
docs = splitter.split_documents(documents)
    

# print(len(docs))
# print(docs)
# print(type(docs))

# embedding 생성
from langchain_community.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')


# chroma db에 vector 저장
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(documents=docs, embedding=embeddings)

# retriever 생성
retriever = db.as_retriever()

# llm 초기화
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

# 질의 응답 체인 생성 (Runnable 인터페이스 사용)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""다음 문맥을 기반으로 질문에 답하세요.
문맥: {context}
질문: {question}
답변:""")

chain = {"context": retriever,
         "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


# 질의 응답 체인 실행
query = "인공지능이란 무엇일까?"
answer = chain.invoke(query)

print(answer)