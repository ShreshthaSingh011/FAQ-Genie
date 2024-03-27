from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.llms import GooglePalm
load_dotenv()
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5)

# To create embeddings and vector database
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

instructor_embeddings = HuggingFaceInstructEmbeddings()
vector_db_file_path ="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vector_db_file_path)

# We won't be using vector db in memory, because it takes too much time
# We don't want to execute it everytime, when we launch our streamlit
#thats why we'll save this vectordb to a disk
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vector_db_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
   chain = get_qa_chain()
   print(chain("EMI options"))

