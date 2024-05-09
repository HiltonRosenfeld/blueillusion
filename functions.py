from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

import streamlit as st

from config import Config
import base64
from PIL import Image
import io

import os
#from dotenv import load_dotenv

#load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_VECTOR_ENDPOINT = os.getenv("ASTRA_VECTOR_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_VECTOR_ENDPOINT = Config.ASTRA_VECTOR_ENDPOINT
ASTRA_DB_KEYSPACE = Config.ASTRA_DB_KEYSPACE
ASTRA_DB_COLLECTION = Config.ASTRA_DB_COLLECTION

os.environ["LANGCHAIN_PROJECT"] = "imagesearch"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# Cache Chat Model for future runs
@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model: str = Config.VISION_MODEL):
    print(f"load_model: {model}")
    return ChatOpenAI(
        temperature=0.2,
        model=model,
        streaming=True,
        verbose=True,
        #max_tokens=Config.MAX_TOKENS,
    )

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    print(f"load_vectorstore: {ASTRA_VECTOR_ENDPOINT} / {ASTRA_DB_KEYSPACE} / {ASTRA_DB_COLLECTION}")

    embedding = load_embedding()
    # Get the load_vectorstore store from Astra DB
    return AstraDBVectorStore(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )
    
# Cache Retriever for future runs
@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    print("load_retriever")

    vectorstore = load_vectorstore()
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": Config.TOP_K_VECTORSTORE}
    )




def encode_image(image_path, quality="low"):
    if quality == "full":
        # Full resolution version
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    else:
        # Low resolution version
        with Image.open(image_path) as img:
            # Resize the image to maximum dimension of 512 pixels
            img.thumbnail((512, 512))
            print(f"img.size: {img.size}")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')


# Use an LLM to describe the image
def analyse_image_file(file):
    encoded_string = encode_image(file)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage( content=(Config.SYSTEM_PROMPT)),
            HumanMessage(
                content=[
                    {"type": "text", "text": Config.USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}",
                            #"detail": Config.DETAIL,
                        },
                    },
                ]
            )
        ]
    )
    model = load_model()
    chain = prompt | model
    response = chain.stream({'question': Config.USER_PROMPT,})

    return response



# Use an LLM to describe the image
def search_similar_images(description, user_prompt):
    
    template = "You're a helpful fashion assistant tasked to help users shopping for clothes, shoes and accessories.\n"
    #template += "You like to help a user find the perfect outfit for a special occasion.\n"
    #template += "You should also suggest other items to complete the outfit.\n"
    template += "You will be given a description of fashion items and you will need to find similar items from the context provided.\n"
    #template += "Do not include any information other than what is provied in the context below.\n"
    template += "Include an image of the product taken from the img attribute in the metadata.\n"
    template += "Include the price of the product if found in the context.\n"
    template += "Include a link to buy each item you recommend if found in the context. Here is a sample buy link:\n"
    template += "Buy Now: [Product Name](https://www.blueillusion.com/product-name)\n"
    #template += "If you don't know the answer, just say 'I do not know the answer'.\n"
    template += "If the user has not asked a question related to fashion and Blue Illusion products, you can respond with 'I do not have the products you asked for' and suggest a simialr alternative from BlueIllusion context.\n"
    template += """
Use the following context to answer the question:
{context}

Question:
{question}

Answer in English"""

    prompt = ChatPromptTemplate.from_messages([("system", template)])
    retriever = load_retriever()
    model = load_model()
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['description']),
        'question': lambda x: x['question'],
        #'description': lambda x: x['description'],
    })
    chain = inputs | prompt | model
    response = chain.stream({'question': user_prompt, 'description': description,})

    return response



# handles stream response back from LLM
def stream_parser(stream):
    for chunk in stream:
        #yield chunk["response"]                # for Ollama
        #yield chunk.choices[0].delta.content   # for openAI direct
        yield chunk.content                     # for openAI via Langchain


