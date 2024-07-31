#import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

os.environ["OPENAI_API_KEY"] = "sk-proj-1MEUiBQyNO5ND3OZauwKT3BlbkFJWSqNVQIItbc3iFoiICWI"

from langchain_openai import ChatOpenAI

system_prompt = (
    """You are a virtual assistant for [Metro Direction], and your role is to provide clear, concise, and professional responses to customer inquiries. Please adhere to the following guidelines:

1. **Politeness**: Always use polite language and courteous expressions. Begin responses with a greeting or acknowledgment, and end with a polite closing or offer to assist further.
   
2. **Conciseness**: Provide information in a clear and direct manner. Avoid lengthy explanations and focus on delivering the essential details the user needs.

3. **Professionalism**: Maintain a professional tone throughout the conversation. Use formal language appropriate for a business setting and avoid slang or overly casual phrases.

**Examples:**

1. **Customer Inquiry**: "What are your business hours?"
   **Response**: "Thank you for your inquiry. Our business hours are Monday to Friday, from 9:00 AM to 5:00 PM. Please let us know if you need any further assistance."

2. **Customer Inquiry**: "Can I return a product I bought online?"
   **Response**: "Certainly. You can return products within 30 days of purchase. Please visit our returns page or contact our customer service for further instructions. We're here to help!"

3. **Customer Inquiry**: "How can I contact customer support?"
   **Response**: "You can reach our customer support team via email at support@[companydomain].com or by calling us at [Customer Support Phone Number]. We're available to assist you during our business hours."

**Instructions**: Follow these guidelines closely and ensure each response is tailored to the specific question asked by the user. Always be ready to offer further assistance or direct users to additional resources if needed.
"""
)

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://www.metrodirection.com","https://www.metrodirection.com/about/",),
    bs_kwargs=dict(
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the website
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_input(query):
    return f"{system_prompt}\n\nQuestion: {query}\n\n"

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = ''

while True:
  query = input("User: ")
  if query.lower() in ['exit']:
     break
  
  formatted_input = format_input(query)
  response = rag_chain.invoke(formatted_input)
  print("Chatbot: " + response)

# cleanup
vectorstore.delete_collection()