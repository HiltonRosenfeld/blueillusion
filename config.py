class Config:
    ASTRA_VECTOR_ENDPOINT = "https://8687c3e7-66c5-4679-9c59-b405038cfec2-us-east1.apps.astra.datastax.com"
    ASTRA_DB_KEYSPACE = "blueillusion"
    ASTRA_DB_COLLECTION = "catalogue_img_desc"

    EMBEDDING_MODEL = "text-embedding-3-small"
    VISION_MODEL = "gpt-4-turbo"
    MAX_TOKENS = 300
    DETAIL = "high"

    TOP_K_VECTORSTORE = 8
    TOP_K_MEMORY = 4

    SYSTEM_PROMPT = "You are a fashion consultant and you have been asked to describe the clothing in an image."
    USER_PROMPT = "Describe in detail what clothing is in the image. Make sure to describe the color, material, and style of the clothing."


    INITIAL_SIDEBAR_STATE = "collapsed"
    #INITIAL_SIDEBAR_STATE = "expanded"
