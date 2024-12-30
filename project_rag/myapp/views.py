from django.shortcuts import render, HttpResponseRedirect
from .models import Store_pdf
import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA

# Create your views here.

def Index(request):
    try:
        data = Store_pdf.objects.all().values_list()
        file_path=f'media/{data[len(data)-1][1]}'
        print(file_path)
        response = "Ask Me Something From Your Uploaded PDF"
        if request.method == "POST":
            query = request.POST.get("query")
            # pdf = request.FILES.get('pdf')
            response = "Wait For It Loading"
            # Store_pdf.objects.create(uploaded_file=pdf)
            data = Store_pdf.objects.all().values_list()
            file_path=f'media/{data[len(data)-1][1]}'
            print(file_path)
            credentials = Credentials(
                url="https://us-south.ml.cloud.ibm.com",
                api_key="2WnQDeUOL_P4D3Hc6oOmT5o3yXubHzTtQh1qyJnqk_31",
            )
            project_id = os.environ.get("PROJECT_ID", "dbc08080-24a7-454f-9dcb-eaaf7c9623c8")
        
            # Initialize API client
            api_client = APIClient(credentials=credentials, project_id=project_id)
        
            # Read and process file
            
            file_content = ""
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        try:
                            # Attempt to extract text using UTF-8
                            file_content += page.extract_text()
                        except UnicodeDecodeError:
                            # If UTF-8 fails, try 'latin-1'
                            file_content += page.extract_text(encoding='latin-1')
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as text_file:
                    file_content = text_file.read()
            else:
                raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")

            data = file_content
            print(f"Uploaded file processed: {len(data)} characters")
        
            # Split the text into smaller chunks for vectorization
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=0,
                length_function=len,
            )
            texts = text_splitter.split_text(data)
            print(f"Text split into {len(texts)} chunks.")
        
            # Convert the text chunks into Document objects
            documents = [Document(page_content=text) for text in texts]
        
            # Initialize Watsonx embeddings
            embeddings = WatsonxEmbeddings(
                model_id="ibm/slate-30m-english-rtrvr",
                url=credentials["url"],
                apikey=credentials["apikey"],
                project_id=project_id,
            )
        
            # Create a vector store using Chroma
            docsearch = Chroma.from_documents(documents, embeddings)
            print("Embeddings generated and stored in Chroma.")
        
            # Define model type and parameters for Watsonx Granite model
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
        
            # Build RetrievalQA using the vector store and Granite model
            qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())
            print("Question answering system initialized.")
        
            # Example query
            query = query
            response = qa.invoke(query)
            response = response['result']
            print(f"Answer: {response}")
        return render(request,'index.html',{'data':response,'file':file_path})
    except:
        error = "Error Might Token Quota Reached, Request of 1 token(s) from quota was rejected"
        return render(request,'index.html',{'data':error})


def Upload_Pdf(request):
    if request.method == "POST":
        pdf = pdf = request.FILES.get('pdf')
        Store_pdf.objects.create(uploaded_file=pdf)
        data = Store_pdf.objects.all().values_list()
        # print(data[len(data)-1][1])
        return HttpResponseRedirect('/')
    return render(request,'upload_pdf.html',{'wow':'PDF Uploaded Successfully...'})


def Base(request):
    return render(request, 'base.html')
