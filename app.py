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
    st.set_page_config(page_title="CV")
    st.header("Ask about cv")

    cv = st.file_uploader("Upload a CV", type="pdf", accept_multiple_files=True)

    key = st.text_input('Enter your OPENAI_API_KEY: ', type='password')

    if cv is not None:
        all_text = ""

        for file in cv:
            cv_reader = PdfReader(file)

            text = ""
            for page in cv_reader.pages:
                text += page.extract_text()

            all_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(all_text)

        embeddings = OpenAIEmbeddings(api_key=key).embed_documents(chunks)

        if embeddings:
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            user_question = st.text_input('Ask a question:')
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(api_key=key)

                chain = load_qa_chain(llm, chain_type="stuff")

                inputs = {"input_documents": docs, "question": user_question}
                response = chain.invoke(inputs)

                st.write(response['output_text'])
        else:
            st.write("No embeddings generated. Please check your API key or input text.")

if __name__ == "__main__":
    main()