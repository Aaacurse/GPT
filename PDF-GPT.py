import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
import pickle
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main():
    load_dotenv()
    
    #api_key=os.getenv("GOOGLE_API_KEY")
    api_key=st.secrets['api_key']
    
    st.title("PDF-GPT💭")
    st.subheader("Chat-GPT for your PDFs📄🗣️")
    print(api_key)
    with st.sidebar:
        pdf=st.file_uploader("Upload your PDF file📄",type='pdf')
        "LLM: [Google Gemini Pro](https://python.langchain.com/docs/integrations/llms/google_ai)"
        "Embeddings: [GoogleGenerativeAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai)"
        "Vector Store: [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)"
        "Built with Langchain 🦜🔗"
    
    query=st.text_input("Enter your Query:")
    submit=st.button("Submit")

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        text_spiltter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len 
        ) 
        chunks=text_spiltter.split_text(text=text)
        
        embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')

        vector_store=FAISS.from_texts(chunks,embeddings)
        file_name=pdf.name[:-4]
        
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl","rb") as f:
                vector_store=pickle.load(f)
            #st.write("Embeddings Loaded Sucessfully")
        else:
            with open(f"{file_name}.pkl","wb") as f:
                pickle.dump(vector_store,f)
            #st.write("Embeddings Computed Sucessfully")

        if query is not None:
            if submit:
                with st.spinner("Processing"):
                    docs=vector_store.similarity_search(query=query)
                    #st.write(docs)
                    llm=GoogleGenerativeAI(model="gemini-pro",google_api_key=api_key,temperature=0.7)
                    chain=load_qa_chain(llm=llm,chain_type="stuff")
                    response=chain.run(input_documents=docs,question=query)
                    st.write(response)
                    





if __name__=="__main__":
    main()
