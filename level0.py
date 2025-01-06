
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import time

st.header('Static Chat CrustData', divider='rainbow')


import os
from dotenv import load_dotenv
load_dotenv()



OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
def run_level0():
    try:
        db = FAISS.load_local("crustdata_db", embedding_model, allow_dangerous_deserialization=True)
    except AssertionError as e:
        st.error("Error loading FAISS index. Ensure the embedding model used for indexing matches the one used for querying.")
        raise e

    prompt1 = """You are a Crustdata API expert helping developers understand and implement Crustdata's API.

        Context: {context}


        Most Important: 
        1. Give the Curl / Python commands wherever you think you can give wrt query.
        2. Give elaborative response, as if you are making understand a baby.

        Please address the following:
        1. The endpoint's purpose, features, and authentication details.
        2. Key implementation details like parameters and response structure.
        3. Best practices, usage limits, and performance tips.
        4. Common errors, troubleshooting, and limitations.
        5. Always try to give the source where you are refering(see if there's any link, python documentation, curl command)

        Always try to give CrustDATA API reference.

        Use only the information from the context and mention if anything is missing."""


    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt1),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.9,
        max_tokens=1024,
        api_key=OPENAI_API_KEY
    )

    db_final_retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_final_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )


    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    input_prompt = st.chat_input("Ask a question about the CrustData API")

    if input_prompt:
        with st.chat_message("user"):
            st.write(input_prompt)

        st.session_state.messages.append({"role": "user", "content": input_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                try:
                    result = qa({"question": input_prompt})
                except AssertionError as e:
                    st.error("Error during the retrieval process. Ensure the embedding dimensions match the FAISS index.")
                    raise e
                except ValueError as e:
                    st.error("Missing input keys. Ensure the input contains the required keys.")
                    raise e

                message_placeholder = st.empty()
                full_response = "**_This is an AI generated response._** \n\n\n"

                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.button('Reset Chat', on_click=reset_conversation)
