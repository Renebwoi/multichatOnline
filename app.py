# import  streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import speech_recognition as sr
# import pyttsx3


# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text=""
#     # for pdf in pdf_docs:
#     pdf_reader=PdfReader(pdf_docs)
#     # with open(pdf, "rb") as pdf_file:
#         # pdf_reader = PdfReader(pdf_file)
#         # Process the PDF using reader.pages
        
#     for page in pdf_reader.pages:
#         text+=page.extract_text()
#     return text
    
# def get_text_chunks(text):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)#change to 10000 and 1000 respectively
#     chunks=text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
#     vector_store.save_local("faiss_index")#for saving the vectors
    
# def get_conversational_chain():
#     prompt_template=""" 
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     the provided context just say, "answer is not available in this context", provide the details of the wrong answer
#     Context:\n {context}?\n
#     Question:\n {question}?\n
    
#     Answer:
#     """
#     model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
#     prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
#     chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
#     return chain

# def user_input(user_question):
#     engine = pyttsx3.init()
    
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    
#     docs =new_db.similarity_search(user_question)
    
#     chain =get_conversational_chain()
    
#     response = chain(
#         { "input_documents":docs, "question": user_question},
#         return_only_outputs=True)
    
#     print(response)
#     st.write("Reply: ", response["output_text"])
#     engine.say(response["output_text"])
#     engine.runAndWait()
    
# def user_input_speak():
#     st.write("Speak your question:")
    
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         audio_data = recognizer.listen(source)
    
#     try:
#         user_question = recognizer.recognize_google(audio_data)
#         st.write("You said:", user_question)
#         response = user_input(user_question)#to process user input on it
#         st.write("Reply:", response)
#     except sr.UnknownValueError:
#         st.write("Sorry, I couldn't understand what you said.")
#     except sr.RequestError:
#         st.write("Sorry, I couldn't reach the Google API.")


# def main():
#     st.set_page_config("Chat With Learning Materials")
#     st.header("Chat with Learning Materials😎")
    
#     user_question = st.text_input("Ask a Question from the PDF files")
    
#     if user_question:
#         user_input(user_question)
        
#     if st.button("Speak"):#changed here too
#         user_input_speak()
    
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit and Process button")
#         # print(pdf_docs, type(pdf_docs))
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")
    
# if __name__ == "__main__":
#     main()


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import speech_recognition as sr
import pyttsx3

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    the provided context just say, "answer is not available in this context", provide the details of the wrong answer
    Context:\n {context}?\n
    Question:\n {question}?\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    engine = pyttsx3.init()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])
    engine.say(response["output_text"])
    engine.runAndWait()
    
def user_input_speak():
    st.write("Speak your question:")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source)
    
    try:
        user_question = recognizer.recognize_google(audio_data)
        st.write("You said:", user_question)
        response = user_input(user_question)
        st.write("Reply:", response)
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand what you said.")
    except sr.RequestError:
        st.write("Sorry, I couldn't reach the Google API.")

def main():
    st.set_page_config("Chat With Learning Materials")
    st.header("Chat with Learning Materials😎")
    
    user_question = st.text_input("Ask a Question from the PDF files")
    
    if user_question:
        user_input(user_question)
        
    if st.button("Speak"):
        user_input_speak()
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit and Process button")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    
if __name__ == "__main__":
    main()
