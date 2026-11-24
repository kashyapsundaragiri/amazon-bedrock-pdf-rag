import os
import boto3
import streamlit as st
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# Bedrock Client Initialization
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-southeast-2"
)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# Data Ingestion and Preprocessing
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(documents)

# Vector Embedding and Vector Store
def get_vector_store(texts):
    vectorstore_faiss = FAISS.from_documents(texts, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    # Use ChatBedrockConverse for modern Claude models (verify exact inference profile ID in ap-southeast-2)
    return ChatBedrockConverse(
        model="apac.anthropic.claude-opus-4-7",  # or "anthropic.claude-opus-4-7" if direct on-demand is supported
        client=bedrock_client,
        max_tokens=4096,
        temperature=0.5,
    )

def get_deepseek_llm():
    # Use ChatBedrockConverse for DeepSeek models
    return ChatBedrockConverse(
        model="deepseek.v3.2",  # verify exact model/inference profile ID in your Bedrock console
        client=bedrock_client,
        max_tokens=4096,
        temperature=1.0,
    )

prompt_template = """Use the following pieces of context to answer the question at the end. 
Provide a detailed explanation and reasoning of at least 250 words. 
If you don't know the answer, just say that you don't know—do not make up an answer.

<context>
{context}
</context>

Question: {question}
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_from_llm(llm, vectorstore_faiss, question):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    response = qa_chain.invoke({"query": question})
    return response["result"]

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Amazon Bedrock")

    user_question = st.text_input("Ask a question about the PDF documents:")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                texts = data_ingestion()
                get_vector_store(texts)
                st.success("Successfully Completed")

    if st.button("Claude LLM Response"):
        if not user_question:
            st.warning("Please enter a question first.")
        elif not os.path.exists("faiss_index"):
            st.warning("Please click 'Vectors Update' in the sidebar first.")
        else:
            with st.spinner("Processing..."):
                vectorstore_faiss = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_claude_llm()
                response = get_response_from_llm(llm, vectorstore_faiss, user_question)
                st.write(response)
                st.success("Successfully Completed")

    if st.button("Deepseek LLM Response"):
        if not user_question:
            st.warning("Please enter a question first.")
        elif not os.path.exists("faiss_index"):
            st.warning("Please click 'Vectors Update' in the sidebar first.")
        else:
            with st.spinner("Processing..."):
                vectorstore_faiss = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_deepseek_llm()
                response = get_response_from_llm(llm, vectorstore_faiss, user_question)
                st.write(response)
                st.success("Successfully Completed")

if __name__ == "__main__":
    main()