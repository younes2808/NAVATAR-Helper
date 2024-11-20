import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
import nest_asyncio

nest_asyncio.apply()

#################################################################
# Tokenizer
#################################################################

model_name = 'norallm/normistral-7b-warm-instruct'

model_config = transformers.AutoConfig.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

messages = [
    {"role": "user", "content": "Hva er hovedstaden i Norge?"},
    {"role": "assistant", "content": "Hovedstaden i Norge er Oslo. Denne byen ligger i den sørøstlige delen av landet, ved Oslofjorden. Oslo er en av de raskest voksende byene i Europa, og den er kjent for sin rike historie, kultur og moderne arkitektur. Noen populære turistattraksjoner i Oslo inkluderer Vigelandsparken, som viser mer enn 200 skulpturer laget av den berømte norske skulptøren Gustav Vigeland, og det kongelige slott, som er den offisielle residensen til Norges kongefamilie. Oslo er også hjemsted for mange museer, gallerier og teatre, samt mange restauranter og barer som tilbyr et bredt utvalg av kulinariske og kulturelle opplevelser."},
    {"role": "user", "content": "Gi meg en liste over de beste stedene å besøke i hovedstaden"}
]
gen_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

#################################################################
# BitsAndBytes Parameters
#################################################################

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load Pre-trained Model
#################################################################

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

text_generation_pipeline = pipeline(
    model=model,
    task="text-generation",
    tokenizer=tokenizer,
    max_new_tokens=1024,
    top_k=64,  # top-k sampling
    top_p=0.9,  # nucleus sampling
    temperature=0.3,  # a low temperature to make the outputs less chaotic
    repetition_penalty=1.0,  # turn the repetition penalty off
    do_sample=True,  # randomly sample the outputs
    use_cache=True  # speed-up generation
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

#################################################################
# Fetch Documents
#################################################################

DATA_PATH = "NEET"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
print("ANTALL DOKUMENTER: ", len(documents))
chunks = split_documents(documents)

#################################################################
# Check if Milvus Index Exists
#################################################################

from pymilvus import connections, utility

def check_index_exists(collection_name):
    connections.connect(uri="./milvus_demo.db")
    return utility.has_collection(collection_name)

collection_name = "your_collection_name"  # Replace with your collection name

if not check_index_exists(collection_name):
    print("Index does not exist. Creating new embeddings and index.")
    db = Milvus.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
        connection_args={"uri": "./milvus_demo.db"},
        drop_old=False
    )
else:
    print("Index already exists. Using existing index.")
    db = Milvus(
        connection_args={"uri": "./milvus_demo.db"},
        collection_name=collection_name
    )

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

#################################################################
# Create Prompt Template
#################################################################

prompt_template = """
<|im_start|> user
Instruksjon: Du er en assistent som skal svare på spørsmål. Svar på norsk på spørsmålet basert på din kunnskap om muskelplager og smerte. Her er kontekst som kan hjelpe, bruk kun kunnskap fra dette til å svare på spørsmålet:

{kontekst}

Spørsmål:
{spørsmål}<|im_end|>
<|im_start|> assistant
"""

prompt = PromptTemplate(
    input_variables=["kontekst", "spørsmål"],
    template=prompt_template,
)

llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = (
    {"kontekst": retriever, "spørsmål": RunnablePassthrough()}
    | llm_chain
)

while True:
    question = input("Spør mistral 7B: ")

    result = rag_chain.invoke(question)
    print("RESULTAT FRA LLM:")
    print(result)
    print("---", result["text"])
    print("SOURCES:")
    for i in result["kontekst"]:
        print("-->", i.metadata)
