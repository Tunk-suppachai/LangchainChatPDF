
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores  import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI 
from langchain.callbacks import get_openai_callback



def main():
 #   print("Hello World!")
    load_dotenv()
    os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="Abdul PDF", page_icon="🚀")
    st.header("AbdulPDF")

    pdf = st.file_uploader("Upload a pdf", type="pdf")
    if pdf is not None:
        # read pdf
        pdf_read = PdfReader(pdf)
        text = ""
        for page in pdf_read.pages:
            text += page.extract_text()
        # st.write(text)

        # split into chunks
        splitter = CharacterTextSplitter(
            separator= " ", 
            chunk_size= 1000,
            chunk_overlap= 100,
            length_function= len
            )
        chunks = splitter.split_text(text)
        # st.write(chunks)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # ask question
        user_question = st.text_input("Ask your question")
        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=3)
            # st.write(docs)
            chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
            # print the price of the cost
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            st.write(response)



if __name__ == "__main__":
    main()