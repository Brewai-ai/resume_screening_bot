import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain




def main():
    load_dotenv()
   
    st.set_page_config(page_title="Resume_Screening Bot")
    st.header("Resume_Screening Bot")
    

    # upload file
    cv = st.file_uploader("Upload a CV", type="pdf")

    key = st.text_input('Enter your OPENAI_API_KEY: ',type='password')

    # extract the text
    if cv is not None:
        cv_reader = PdfReader(cv)

        text = ""
        for page in cv_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text)
        

        # create embeddings
        embeddings = OpenAIEmbeddings(api_key=key)
        
        knowledge_base = FAISS.from_texts(chunks,embeddings)

        # show user input
        user_question = st.text_input('Ask a question:')
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs) 

           # initialize the language model
            llm = OpenAI(api_key=key)

            # load the question-answering chain
            chain = load_qa_chain(llm, chain_type="stuff")

            # generate a response to the user's question
            inputs = {"input_documents": docs, "question": user_question}
            response = chain.invoke(inputs)

            st.write(response['output_text'])
        

if __name__ == "__main__":
    main()