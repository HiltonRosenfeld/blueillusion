import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import BedrockChat
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.vectorstores.astradb import AstraDB
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_VECTOR_ENDPOINT = os.environ["ASTRA_VECTOR_ENDPOINT_BO"]
ASTRA_DB_KEYSPACE = "blueillusion"
ASTRA_DB_COLLECTION = "catalogue"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

os.environ["LANGCHAIN_PROJECT"] = "blueillusion"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


print("Started")


# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 8
top_k_memory = 3

###############
### Globals ###
###############

global lang_dict
global rails_dict
global embedding
global vectorstore
global retriever
global model
global chat_history
global memory


#######################
### Resources Cache ###
#######################

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    #return OpenAIEmbeddings(model="text-embedding-3-small")
    # Get the Bedrock Embedding
    return BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")

    

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    print(f"load_vectorstore: {ASTRA_DB_KEYSPACE} / {ASTRA_DB_COLLECTION}")
    # Get the load_vectorstore store from Astra DB
    return AstraDB(
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
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

# Cache Chat Model for future runs
@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="openai.gpt-3.5"):
    print(f"load_model: {model_id}")
    # if model_id contains 'openai' then use OpenAI model
    if 'openai' in model_id:
        if '3.5' in model_id:
            gpt_version = 'gpt-3.5-turbo'
        else:
            gpt_version = 'gpt-4-turbo-preview'
        return ChatOpenAI(
            temperature=0.2,
            model=gpt_version,
            streaming=True,
            verbose=True
            )
    # else use Bedrock model
    return BedrockChat(
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id=model_id,
        streaming=True,
        #callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={"temperature": 0.1},
    )

# Cache Chat History for future runs
@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history():
    print("load_chat_history")
    return AstraDBChatMessageHistory(
        session_id=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

# Cache prompt
# 
#Include the price of the product if found in the context.
#Do not include images in your response.
#Provide at most 2 items that are relevant to the user's question.
#You prefer to use bulletpoints to summarize.
#You can also ask clarifying questions if you need more information to answer the user's question.

@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You're a helpful fashion assistant tasked to help users shopping for clothes and accessories. 
You like to help a user find the perfect outfit for a special occasion. 
You should also suggest other items to complete the outfit.
You're friendly and you answer extensively with multiple sentences. 
Focus on the user's needs and provide the best possible answer.
Do not include any information other than what is provied in the context below.
Include an image of the product taken from the img attribute in the metadata (without a label).
Include a link to buy each item you recommend if found in the context. Here is a sample buy link:
Buy Now: [Product Name](https://www.blueillusion.com/product-name)
If you don't know the answer, just say 'I do not know the answer'.
If the user has not asked a question related to clothing and Blue Illusion products, you can respond with 'I do not know the answer' as well.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in English"""

    return ChatPromptTemplate.from_messages([("system", template)])



#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi, I'm your personal shopping assistant!")]


############
### Main ###
############

# Write the welcome text
st.markdown(Path('welcome.md').read_text())

# DataStax logo
with st.sidebar:
    st.image('./public/logo.svg')
    st.text('')

# Initialize
with st.sidebar:
    embedding = load_embedding()
    #model = load_model()
    vectorstore = load_vectorstore()
    retriever = load_retriever()
    chat_history = load_chat_history()
    memory = load_memory()
    prompt = load_prompt()

# Drop the Conversational Memory
with st.sidebar:
    with st.form('delete_memory'):
        st.caption('Delete the history in the conversational memory.')
        submitted = st.form_submit_button('Delete chat history')
        if submitted:
            with st.spinner('Delete chat history'):
                memory.clear()

with st.sidebar:
    # Add a drop down to choose the LLM model
    st.caption('Choose the LLM model')
    model_id = st.selectbox('Model', [
        'openai.gpt-3.5',
        'openai.gpt-4',
        'amazon.titan-text-express-v1',
        'anthropic.claude-v2', 
        'ai21.j2-mid-v1', 
        'meta.llama2-13b-chat-v1',
        ])
    model = load_model(model_id)

st.markdown("<style> img {width: 200px;} </style>", unsafe_allow_html=True,
)

# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
if question := st.chat_input("What's up?"):
    print(f"Got question: \"{question}\"")

    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print("Display user prompt")
    with st.chat_message("user"):
        st.markdown(question)

    # Get the results from Langchain
    print("Get AI response")
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        print(f"Using memory: {history}")

        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        })
        print(f"Using inputs: {inputs}")

        chain = inputs | prompt | model
        print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)]})
        print(f"Response: {response}")
        #print(embedding.embed_query(question))
        content = response.content

        # Write the final answer without the cursor
        response_placeholder.markdown(content)


        # Add the result to memory
        memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))
