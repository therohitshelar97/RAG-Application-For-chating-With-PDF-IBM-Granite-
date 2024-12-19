# !pip install wget
# !pip install -U "langchain>=0.3,<0.4"
# !pip install -U "ibm_watsonx_ai>=1.1.22"
# !pip install -U "langchain_ibm>=0.3,<0.4"
# !pip install -U "langchain_chroma>=0.1,<0.2"
# !pip install PyPDF2

import os
import getpass

from ibm_watsonx_ai import Credentials

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="9JAu6q17I5PwWYISP9HhsqvlygIkbygflVj6fZ74plwm",
)

try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = "97ea408b-8fee-4927-86e3-13fdc670ee98"

from ibm_watsonx_ai import APIClient

api_client = APIClient(credentials=credentials, project_id=project_id)

from IPython.display import display
import ipywidgets as widgets
import PyPDF2

# File upload widget
upload_widget = widgets.FileUpload(accept='.pdf,.txt', multiple=False)

# Function to process uploaded files
def process_uploaded_file(uploaded_file):
    for name, file_info in uploaded_file.value.items():
        file_content = file_info['content']
        if name.endswith('.pdf'):
            # Save and read PDF
            with open(name, 'wb') as f:
                f.write(file_content)
            with open(name, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    try:
                        # Attempt to extract text using UTF-8
                        text += page.extract_text()
                    except UnicodeDecodeError:
                        # If UTF-8 fails, try 'latin-1'
                        text += page.extract_text(encoding='latin-1')
            return text
        elif name.endswith('.txt'):
            # Read text file
            return file_content.decode('utf-8')
        else:
            raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")

# Display the upload widget
display(upload_widget)

# Process the file if uploaded
uploaded_file = upload_widget.value
if uploaded_file:
    data = process_uploaded_file(upload_widget)
    print(f"Uploaded file processed: {len(data)} characters")
else:
    raise ValueError("No file uploaded!")

from langchain.text_splitter import CharacterTextSplitter

# Split the text into smaller chunks for vectorization
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(data)
print(f"Text split into {len(texts)} chunks.")

# !pip install -U "langchain_chroma>=0.1,<0.2"
from langchain_chroma import Chroma
from langchain_ibm import WatsonxEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Watsonx embeddings
embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id
)

# Use RecursiveCharacterTextSplitter for better handling of long text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Set chunk size considering the model's max sequence length
    chunk_overlap=0,
    length_function=len,  # Use len to count characters
)
texts = text_splitter.split_text(data)

# Convert the text chunks into Document objects
documents = [Document(page_content=text) for text in texts]

# Create a vector store for document retrieval
docsearch = Chroma.from_documents(documents, embeddings)
print("Embeddings generated and stored in Chroma.")

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM

# Define model type and parameters
model_id = ModelTypes.GRANITE_13B_CHAT_V2
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

# Initialize the Granite model
watsonx_granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)
print("Watsonx Granite model initialized.")

from langchain.chains import RetrievalQA

# Build RetrievalQA using the vector store and Granite model
qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())
print("Question answering system initialized.")

# Example query
query = "which kind of campaigns sachin had worked ?"
response = qa.invoke(query)
print(f"Answer: {response}")