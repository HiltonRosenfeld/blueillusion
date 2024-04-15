# AI Fashion Shop Assistant

This demo is available to run on [Streamlit Cloud](https://fashionshopassistant.streamlit.app/), or you can run the demo yourself using the instructions below.

Demo Features:
- Keeps track of each user's Conversation History separately. That way one users content does not interfere with any other user.
- Allows the user to choose which LLM Model to use.


## Using the demo

1. Select your preferred LLM.
2. Ask a question.
    - You may be asked for more information to help the assisstant make good recomendations.
3. Delete your conversation history as needed.



## Setup to run your own instance

### 1. Astra DB

1. Create a new or select an existing Astra Keyspace to hold the vector catalogue and the conversation history.
2. Get your Astra DB Token
3. Get your Astra DB API Endpoint

### 2. Load Data
The shopping catalogue has already been crawled from the website and is held in the pickle file website_data.pkl. That file must be read and stored into the vector database, along with embeddings. The Jupyter notebook `embed.ipynb` manages that process, with the ability to start from crawling the website, or to start with the included Pickle file.

#### Process included Pickle file

1. Run the Jupyter notebook from section **Create the Vector Store**

### 3. Secrets and Keys
You have the choice of storing your keys in a `.env` file or in Streamlit Secrets. The format of each is the same.

1. .env file

    1. Copy the `.env.sample` file to `.env`
    2. Provide your keys for the relevant attributes in the file
    3. Set `LOCAL_SECRETS = True` in the app file `app_blueillusion.py`

2. Streamlit secrets

    1. Copy the `.env.sample` file to your streamlit secrets.toml. [See Streamlit docs fro more help](https://docs.streamlit.io/develop/concepts/connections/secrets-management)
    2. Provide your keys for the relevant attributes in the file
    3. Set `LOCAL_SECRETS = False` in the app file `app_blueillusion.py`

### 4. Run the app

    `streamlit run app_blueillusion.py`



