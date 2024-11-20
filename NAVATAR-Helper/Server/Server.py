import socket
import os
import time
import re
import unicodedata
import shutil
import signal
import threading
import textwrap
import sys
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Setup')))
from setupLLM import setupLLM
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
import fitz  # PyMuPDF
from fuzzywuzzy import fuzz

# Lock for synchronizing access to request handling
request_lock = threading.Lock()

# Global variable for server socket
serverSocket = None
fullKontekst = []  # List to hold full context documents
kunContent = []  # List to hold content from the documents
kunMetadata = []  # List to hold metadata from the documents

db = 0  # Database variable for Milvus
mistral_llm = 0  # Variable to hold the language model
prompt = 0  # Variable for the prompt template

def createRagChain():
    """
    Creates a Retrieval-Augmented Generation (RAG) chain that utilizes 
    a language model and a vector database (Milvus) for answering questions 
    based on provided context.

    Returns:
        rag_chain: The RAG chain constructed from the retriever and LLM chain.
    """
    global mistral_llm
    mistral_llm = setupLLM()  # Initialize the language model
    DATABASE_PATH = "./../Database/milvus_512_newpdf.db"
    model_kwargs = {"trust_remote_code": True}
    global db
    # Set up the Milvus database with an embedding function
    db = Milvus(embedding_function=HuggingFaceEmbeddings(model_name='Alibaba-NLP/gte-multilingual-base', model_kwargs=model_kwargs),
                connection_args={"uri": DATABASE_PATH})

    prompt_template = """
    <|im_start|> user
    Instruction: {instruksjon}

    Context:
    {kontekst}

    Question:
    {spørsmål}<|im_end|>
    <|im_start|> assistant
    """

    global prompt
    # Create a prompt template with specified input variables
    prompt = PromptTemplate(
        input_variables=["kontekst", "spørsmål", "instruksjon"],
        template=prompt_template,
    )

    rag_chain = (
        {"kontekst": retriever, "spørsmål": RunnablePassthrough(), "instruksjon": getPromptInstruction}
        | llm_chain  # Combine the retriever and LLM chain
    )
    return rag_chain

@chain
def retriever(query: str) -> List[Document]:
    """
    Retrieves documents from the Milvus database based on a similarity search 
    for the given query.

    Args:
        query (str): The search query string.

    Returns:
        List[Document]: A list of documents retrieved from the database.
    """
    param = {
        "metric_type": "L2",
        "params": {
            "radius": 0.62,
            "range_filter": 0.0
        }
    }
    try:
        docs, scores = zip(*db.similarity_search_with_score(query, k=6, param=param))
    except:
        return []
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score  # Add score to document metadata
    
    global fullKontekst
    global kunContent
    global kunMetadata
    fullKontekst = docs  # Store full context documents
    for doc in docs:
        kunContent.append(doc.page_content)  # Store document content
        kunMetadata.append(doc.metadata)  # Store document metadata

    return kunContent  # Return only the content of the documents

@chain
def llm_chain(inputt):
    """
    Processes the input context and question using the language model (LLM).
    
    Args:
        inputt: A dictionary containing the context and question.

    Returns:
        dict: The response from the LLM, containing the generated text.
    """
    if (len(inputt["kontekst"]) == 0):
        print("\n --> No context found for the question; not initializing LLM")
        return {"text": "No context found."}  # Return an empty response instead of None
    return LLMChain(llm=mistral_llm, prompt=prompt)  # Generate a response using the LLM

def getPromptInstruction(i):
    """
    Provides instructions for the language model assistant.

    Args:
        i: Index to be used if needed for variations (currently unused).

    Returns:
        str: The instruction text for the assistant.
    """
    instruksjon = "You are an assistant that answers questions about NEET. You cannot use prior knowledge, you can only use facts stated in the context below (context is a list contained within square brackets [], between 'Context:' and 'Question:'). If you do not find the answer to the question there, or the list is empty, you must NOT answer the question, but only say verbatim 'I do not know.'"
    return instruksjon  # Return the instruction text

def delete_marked_pdfs():
    """
    Deletes marked PDF files from the specified directory.

    Checks for the existence of the marked PDF directory and deletes 
    all files within it.
    """
    marked_pdf_folder = "./r2/"
    
    # Check if the marked PDF folder exists
    if os.path.exists(marked_pdf_folder):
        # Iterate through all files in the folder
        for filename in os.listdir(marked_pdf_folder):
            file_path = os.path.join(marked_pdf_folder, filename)
            # Check if it is a file before deletion
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted file: {file_path}")
        print(f"Deleted all PDF files in the folder: {marked_pdf_folder}")
    else:
        print("No marked PDF files to delete.")

def signal_handler(sig, frame):
    """
    Handles signals to clean up resources and shut down the server gracefully.

    Args:
        sig: The signal number.
        frame: The current stack frame (unused).
    """
    global serverSocket  # Access the global serverSocket
    print('\nReceived termination signal, cleaning up...')
    delete_marked_pdfs()  # Delete marked PDFs on exit
    if serverSocket:
        serverSocket.close()  # Close the server socket
    print("The server has been shut down.")
    exit(0)  # Exit the program

# Register the signal handler for termination signals
signal.signal(signal.SIGINT, signal_handler)  # For CTRL+C
signal.signal(signal.SIGTERM, signal_handler)  # For process termination

def convert_sources_to_links(response):
    """
    Converts source references in the response text into clickable links.

    Args:
        response (str): The response text containing source references.

    Returns:
        str: The response text with converted links for sources.
    """
    pdf_base_url = "https://rag.cs.oslomet.no/r2/marked_"
    pdf_pattern = r"Page: (\d+), ([\w\-. &(),'+@~]+\.pdf)"

    def replace_with_link(match):
        page_number = match.group(1)
        pdf_file = match.group(2)
        return f"Page: {page_number}, [{pdf_file}]({pdf_base_url}{pdf_file}#page={page_number})"

    response = re.sub(pdf_pattern, replace_with_link, response)  # Replace matches with links
    return response

def mark_pdf(pdf_name, page_number, chunk_text, similarity_threshold=75):
    """
    Marks a chunk of text in a specified PDF by highlighting it.

    Args:
        pdf_name (str): The name of the PDF file.
        page_number (int): The page number where the text should be marked.
        chunk_text (str): The text chunk to be marked in the PDF.
        similarity_threshold (int): The threshold for text similarity matching (default is 75).

    Returns:
        str: The path to the marked PDF file if successful, None otherwise.
    """
    try:
        # Specify the file path where the PDF is located
        pdf_folder = "./../NEET-PDFs/"
        pdf_path = os.path.join(pdf_folder, pdf_name)

        # Open the PDF file
        doc = fitz.open(pdf_path)

        # Go to the specified page (page numbers start from 0 in PyMuPDF)
        page = doc.load_page(page_number - 1)

        # Function to remove hyphens at line ends and replace newlines with spaces
        def normalize_text(text):
            text = re.sub(r'-\n', '', text)  # Remove hyphens at line ends
            text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
            return ' '.join(text.split())  # Remove extra whitespace

        # Retrieve and normalize all text on the page
        page_text = page.get_text("text")
        normalized_page_text = normalize_text(page_text)

        # Normalize the chunk text
        normalized_chunk = normalize_text(chunk_text)

        # Function to check exact sequence with flexibility in whitespace/punctuation
        def find_exact_sequence(chunk, page_text, threshold):
            chunk_len = len(chunk)
            for i in range(len(page_text) - chunk_len + 1):
                # Check a subsequence of the correct length
                segment = page_text[i:i + chunk_len]
                # Fuzzy matching to handle punctuation and whitespace
                if fuzz.ratio(chunk, segment) >= threshold:
                    return segment  # Return the matching segment
            return None  # Return None if no match found

        # Search for the first exact sequence that resembles the chunk
        matched_segment = find_exact_sequence(normalized_chunk, normalized_page_text, similarity_threshold)
        
        if matched_segment:
            # Find the position(s) of the matched segment in the PDF and highlight it
            text_instances = page.search_for(matched_segment)
            for inst in text_instances:
                page.add_highlight_annot(inst)  # Highlight the matched text

            # Specify the folder for marked PDFs
            marked_pdf_folder = "./r2/"

            # Check if the folder exists, create it if not
            if not os.path.exists(marked_pdf_folder):
                os.makedirs(marked_pdf_folder)

            # Save the marked PDF file in the "marked_pdf" folder
            output_pdf_name = f"marked_{pdf_name}"
            output_pdf_path = os.path.join(marked_pdf_folder, output_pdf_name)
            doc.save(output_pdf_path)  # Save the edited PDF

            # Close the PDF file after editing
            doc.close()

            print(f"The text chunk was marked in the PDF: {output_pdf_name}")
            return output_pdf_path  # Return the path to the marked PDF
        else:
            print(f"The text chunk '{chunk_text}' was not found on page {page_number}.")
            doc.close()
            return None  # Return None if the text chunk was not found
    except Exception as e:
        print(f"An error occurred while marking the PDF: {e}")
        return None  # Return None in case of an exception


def handle_client_requests():
    """
    Sets up a server to handle incoming client requests, processes those requests 
    using the RAG chain, and sends back responses.

    This function listens for incoming connections, handles each request 
    by invoking the RAG chain, and manages the marking of PDF documents.
    """
    global serverSocket
    serverPort = 8501  # Port number for the server
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(('', serverPort))  # Bind the socket to the port
    serverSocket.listen(1)  # Start listening for incoming connections

    rag_chain = createRagChain()  # Create the RAG chain

    print(f"\nServer is ready and listening on port {serverPort}...")

    try:   
        while True:
            clientSocket, clientAddress = serverSocket.accept()  # Accept a client connection
            
            with request_lock:
                try:
                    delete_marked_pdfs()  # Delete marked PDFs before handling the request
                    message = clientSocket.recv(2048).decode()  # Receive the client's message
                    print(f"\nReceived question: {message}")

                    start_time = time.time()  # Start timing the processing

                    # Get the response from the RAG chain
                    result = rag_chain.invoke(message)

                    global kunContent
                    global kunMetadata

                    if len(kunContent) == 0:  # No context available
                        response_message = "I'm afraid I can't answer this question. Please ask me about NEET."
                    else:
                        sources = []
                        for i, doc in enumerate(kunMetadata):
                            # Retrieve and clean text chunk
                            chunk_text = (kunContent[i])  # Clean the chunk
                            page_number = doc.get('page', 0) + 1  # Get the page number (1-based)
                            source_file = os.path.basename(doc['source'])  # Get the file name from source

                            # Try to mark the PDF with the cleaned text chunk
                            marked_pdf_path = mark_pdf(source_file, page_number, chunk_text)

                            if marked_pdf_path:
                                sources.append(f"Page: {page_number}, {source_file}, Source text: '{chunk_text}'")
                            else:
                                sources.append(f"Page: {page_number}, {source_file}, Source text: '{chunk_text}' (Unable to mark)")

                        # Process the response from the chain
                        answer = result["text"].split("<|im_start|> assistant")[-1].strip()
                        answer = textwrap.dedent(answer).strip()  # Clean the answer
                        answer = "\n".join(line.lstrip() for line in answer.splitlines())  # Remove leading spaces

                        # Format the sources
                        sources_text = "\n".join(sources)
                        formatted_sources = convert_sources_to_links(sources_text)  # Convert sources to links
                        response_message = f"{answer}\n\nSources:\n{formatted_sources}"

                    elapsed_time = time.time() - start_time  # Calculate elapsed time
                    print(f"\nTime taken to process the request: {elapsed_time:.2f} seconds")
                    print(f"\nSending response to client: {response_message}")

                    # Send the response back to the client
                    bytes_sent = 0
                    while bytes_sent < len(response_message):
                        bytes_sent += clientSocket.send(response_message.encode()[bytes_sent:])

                except Exception as e:
                    print(f"An error occurred: {e}")
                    clientSocket.send("An error occurred while processing the request.".encode())
                
                finally:
                    kunContent = []  # Reset the content list
                    kunMetadata = []  # Reset the metadata list
                    clientSocket.close()  # Close the client socket

    except Exception as e:
        print(f"The server encountered an error: {e}")

    finally:
        serverSocket.close()  # Close the server socket

if __name__ == "__main__":
    handle_client_requests()  # Start handling client requests
