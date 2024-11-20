import socket
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
import re
import torch
import time
import inspect
import ast
import nest_asyncio
import torch
import transformers
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  pipeline
)

import gc
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

import nest_asyncio

# Updated evaluation prompt without checking the source
EVAL_PROMPT = """
<|im_start|> user
Instruksjon:
Vennligst svar på dette spørsmålet.
Spørsmål: {question}
<|im_end|>
<|im_start|> assistant
"""
def setupLLM():

    """
    Sets up a Language Model (LLM) for Norwegian text generation using a pre-trained model 
    with quantization for efficient GPU usage. The function returns a configured pipeline for text generation.

    Returns:
        mistral_llm: A HuggingFacePipeline object ready for text generation.
    """

    #################################################################
    # Tokenizer Setup
    #################################################################

    ## In this section, the specific language model (LLM) is chosen.
    ## If using a different model, the model name can be updated. 
    ## For example, for the Norwegian GPT model: model_name='NbAiLab/nb-gpt-j-6B'.
    model_name = 'norallm/normistral-7b-warm-instruct'  # Model selected for this task

    # Load the model configuration for the specified model.
    model_config = transformers.AutoConfig.from_pretrained(model_name)

    # Load the tokenizer for the model. The tokenizer is responsible for converting text into tokens that the model can understand.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token to be the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure padding to occur on the right side, as per the model's requirements.
    tokenizer.padding_side = "right"

    # Set up the chat template specific to 'norallm/normistral-7b-warm-instruct'.
    # Different models may require different template setups.
    messages = [
        {"role": "user", "content": "Hva er hovedstaden i Norge?"},  # Example question from the user
        {"role": "assistant", "content": "Hovedstaden i Norge er Oslo..."},  # Example response from the assistant
        {"role": "user", "content": "Gi meg en liste over de beste stedene å besøke i hovedstaden"}  # Follow-up question
    ]
    
    # Prepare the input data for the model by applying the chat template.
    gen_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    #################################################################
    # BitsandBytes Parameters for Efficient Model Loading
    #################################################################

    # Enable 4-bit precision for faster and more memory-efficient model loading.
    use_4bit = True

    # Set the compute data type for 4-bit base models.
    bnb_4bit_compute_dtype = "float16"

    # Set quantization type to nf4, which specifies a specific quantization method.
    bnb_4bit_quant_type = "nf4"

    # Optionally, enable nested quantization (double quantization) for even smaller model size.
    use_nested_quant = False

    #################################################################
    # Quantization Configuration Setup
    #################################################################

    # Set the compute dtype for the model based on the chosen precision.
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Create the configuration for the BitsandBytes quantization process.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check if the GPU supports bfloat16 precision for optimized training.
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    #################################################################
    # Load Pre-trained Model for Text Generation
    #################################################################

    # Load the pre-trained model using the configuration and the quantization setup.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # Apply the quantization settings during model loading
    )
    
    ## If switching to a different LLM, ensure the correct configuration parameters are used.
    ## For instance, these parameters are tailored to 'normistral-7b-warm-instruct': https://huggingface.co/norallm/normistral-7b-warm-instruct
    
    # Set up a text generation pipeline using the loaded model, specifying generation parameters.
    text_generation_pipeline = pipeline(
        model=model,
        task="text-generation",  # Task type: text generation
        tokenizer=tokenizer,
        max_new_tokens=1024,  # Maximum length of generated tokens
        top_k=64,  # Top-k sampling for selecting the next token
        top_p=0.9,  # Nucleus sampling for more diverse output
        temperature=0.3,  # Low temperature to make the outputs more deterministic and less chaotic
        repetition_penalty=1.0,  # Disable repetition penalty
        do_sample=True,  # Enable random sampling to introduce diversity
        use_cache=True  # Enable caching for faster inference
    )

    # Wrap the pipeline with HuggingFacePipeline to integrate it into the system.
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    ## Unknown use of nest_asyncio.apply(), possibly related to running asynchronous code in a blocking environment.
    nest_asyncio.apply()

    return mistral_llm  # Return the LLM for use in evaluation or further tasks

# Read the contents of the RAG_results.txt file and evaluate each question
def fase3():
    """
    This function reads a list of questions from a file, evaluates each question 
    using a large language model (LLM) to generate answers, and stores the results.
    It also measures the time taken for each evaluation and calculates the average response time.
    """

    # Initialize an empty list to store questions from the file
    questions = []

    # Open the 'questions.txt' file to read the questions
    with open('questions.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()  # Read all lines from the file
        
        # Iterate through each line in the file to extract questions
        for line in lines:
            # Check if the line starts with "Question:" indicating it's a question
            if line.startswith("Question:"):
                # Extract the question text by removing the "Question:" prefix and any extra whitespace
                question = line[len("Question:"):].strip()
                questions.append(question)  # Add the question to the list

    # Initialize the LLM model using the setupLLM function
    mistral_llm = setupLLM()

    count = 1  # Counter for numbering the questions
    tid = 0  # Variable to accumulate total time taken for all responses

    # Iterate through each question to generate and evaluate an answer
    for question in questions:     
        
        # Prepare the prompt for evaluation using the predefined EVAL_PROMPT template
        prompt = EVAL_PROMPT.format(
            question=question,  # The question to be evaluated
        )

        # Create an evaluation chain to check if the response from the LLM matches the expected answer
        eval_chain = LLMChain(llm=mistral_llm, prompt=PromptTemplate(input_variables=["question"], template=EVAL_PROMPT))

        # Run the evaluation chain and handle possible runtime errors such as memory issues
        try:
            starttid = time.time()  # Record the start time for this question
            eval_result = eval_chain.run({
                "question": question  # Pass the question to the LLM for evaluation
            })
            slutttid = time.time()  # Record the end time for this question

            # Calculate the time taken to process the current response and add it to the total time
            response_time = slutttid - starttid
            tid += response_time

        except RuntimeError as e:
            # Handle out-of-memory errors for the GPU (CUDA) by retrying the question
            if 'CUDA out of memory' in str(e):
                print("testen feilet ...")  # Print an error message if memory issue occurs
                time.sleep(10)  # Pause before retrying the question
                gc.collect()  # Trigger garbage collection to free up memory
                continue  # Skip this question and move to the next

        # Clean the evaluation result by stripping any extra text and making it lowercase
        eval_result_cleaned = eval_result.split("<|im_start|> assistant")[-1].strip()
        eval_result_cleaned = eval_result_cleaned.lower()  # Convert the result to lowercase for uniformity
        
        # Format the result string to include the question number and answer from the LLM
        result_entry = f"Question Number: {count}:\n"
        result_entry = f"Question : {question}:\n"
        result_entry += f"Answer from LLM without RAG: {eval_result_cleaned}\n"
        
        count += 1  # Increment the question counter
        
        # Append the result entry to the 'Answer_without_RAG.txt' file
        with open("Answer_without_RAG.txt", "a", encoding='utf-8', errors='replace') as llm_file:
            llm_file.write(result_entry + "\n")

    # Calculate the average response time across all questions
    gjennomsnitt = tid / count
    
    # Prepare the string for the average response time
    tid = ""
    tid += f"gjennomsnittlig responstid: {gjennomsnitt}\n"

    # Append the average response time to the 'Answer_without_RAG.txt' file
    with open("Answer_without_RAG.txt", "a", encoding='utf-8', errors='replace') as llm_file:
        llm_file.write(tid + "\n")

        
if __name__ == "__main__":
    fase3()  # Execute the fase3 function
