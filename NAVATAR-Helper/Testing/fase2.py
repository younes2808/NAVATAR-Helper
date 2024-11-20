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

# Evaluation prompt for checking for matching response
EVAL_PROMPT = """
<|im_start|> user
Instruksjon:
Vurder om det faktiske svaret samsvarer med det forventede svaret. Hvis de samsvarer, svar med "sant" eller "samsvarende". Hvis de ikke samsvarer, svar med "usant" eller "ikke samsvarende". Svar kort og presist.

Forventet svar: {expected_response}
Faktisk svar: {actual_response}

Svar med enten 'sant' (true) eller 'usant' (false).
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
    # Tokenizer
    #################################################################

    # Specify the pre-trained model to be used.
    # Example: For using nb-gpt, replace with: model_name='NbAiLab/nb-gpt-j-6B'
    model_name = 'norallm/normistral-7b-warm-instruct'

    # Load the configuration for the specified model
    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
    )

    # Initialize the tokenizer. This will handle text input and output formatting for the model.
    # The `trust_remote_code=True` allows loading custom model/tokenizer implementations.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the same as the end-of-sequence token.
    tokenizer.padding_side = "right"  # Align tokens to the right during padding.

    # Example chat template for the `norallm/normistral-7b-warm-instruct` model.
    # If using a different model, the format of the template may need to be adjusted.
    messages = [
        {"role": "user", "content": "Hva er hovedstaden i Norge?"},
        {"role": "assistant", "content": "Hovedstaden i Norge er Oslo. Denne byen ligger i den sørøstlige delen av landet, ved Oslofjorden. Oslo er en av de raskest voksende byene i Europa, og den er kjent for sin rike historie, kultur og moderne arkitektur. Noen populære turistattraksjoner i Oslo inkluderer Vigelandsparken, som viser mer enn 200 skulpturer laget av den berømte norske skulptøren Gustav Vigeland, og det kongelige slott, som er den offisielle residensen til Norges kongefamilie. Oslo er også hjemsted for mange museer, gallerier og teatre, samt mange restauranter og barer som tilbyr et bredt utvalg av kulinariske og kulturelle opplevelser."},
        {"role": "user", "content": "Gi meg en liste over de beste stedene å besøke i hovedstaden"}
    ]
    # Prepare the input using the tokenizer's chat template function
    gen_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    #################################################################
    # Bits and Bytes (bnb) Parameters for Efficient Model Loading
    #################################################################

    # Enable 4-bit precision loading for memory efficiency
    use_4bit = True

    # Set the computation data type for 4-bit precision
    bnb_4bit_compute_dtype = "float16"

    # Choose the quantization type: fp4 or nf4 (nf4 is generally more robust for LLMs)
    bnb_4bit_quant_type = "nf4"

    # Enable or disable nested quantization (double quantization)
    use_nested_quant = False

    #################################################################
    # Quantization Configuration Setup
    #################################################################

    # Convert the string data type to the corresponding PyTorch type
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Configure the quantization settings using BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,  # Use 4-bit quantization
        bnb_4bit_quant_type=bnb_4bit_quant_type,  # Specify the quantization type
        bnb_4bit_compute_dtype=compute_dtype,  # Specify the compute data type
        bnb_4bit_use_double_quant=use_nested_quant,  # Enable/disable nested quantization
    )

    # Check GPU compatibility for bfloat16 precision
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:  # GPUs with compute capability >= 8 support bfloat16
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    #################################################################
    # Load the Pre-Trained Model
    #################################################################

    # Load the model using the specified configuration and quantization settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # Apply Bits and Bytes quantization
    )
    
    # Configure generation parameters. These settings can significantly affect model behavior:
    #   - top_k: Limits sampling to the top k most probable tokens.
    #   - top_p: Enables nucleus sampling, considering only tokens whose cumulative probability >= top_p.
    #   - temperature: Controls randomness in token selection; lower values make output more deterministic.
    #   - repetition_penalty: Penalizes repeated phrases; set to 1.0 to disable.
    text_generation_pipeline = pipeline(
        model=model,
        task="text-generation",
        tokenizer=tokenizer,
        max_new_tokens=1024,  # Maximum number of tokens the model generates in response
        top_k=64,  # Limit to the top 64 most probable tokens
        top_p=0.9,  # Nucleus sampling threshold
        temperature=0.1,  # Low temperature for more deterministic outputs
        repetition_penalty=1.0,  # Turn off repetition penalty for more natural outputs
        do_sample=True,  # Enable sampling for more varied responses
        use_cache=True  # Cache attention computations for faster inference
    )

    # Wrap the pipeline into a HuggingFacePipeline object for easier integration
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Apply nested asyncio to allow nested event loops (useful in async environments)
    nest_asyncio.apply()

    # Return the configured LLM pipeline
    return mistral_llm


# Function to process and analyze results stored in the RAG_results.txt file
def fase2():

    """
    Executes all predefined test functions, processes their outputs, and evaluates the 
    performance of a Retrieval-Augmented Generation (RAG) pipeline.

    The function performs the following:
        - Collects expected_responses and expected_sources from Phase 1
        - The LLM judges whether the results from the RAG-pipeline match the expected testcases
        - If-statement checks for words that give the result
        - Logs results, including the average response time, to a results file.

    Returns:
        Nothing
    """

    #################################################################
    # Read the contents of the RAG_results.txt file
    #################################################################

    # Open the RAG_results.txt file in read mode and handle encoding errors
    with open("RAG_results.txt", 'r', encoding='utf-8', errors='replace') as file:
        input_text = file.read()

    # Split the input text into sections, where each section corresponds to a question's result
    # Each section starts with the string "Question Number: "
    sections = input_text.strip().split("Question Number: ")[1:]

    #################################################################
    # Initialize lists to store responses and sources for each question
    #################################################################

    # Create empty lists to hold the responses and their corresponding sources for each question
    responses = []
    sources = []

    #################################################################
    # Extract results and sources for each question
    #################################################################

    # Iterate over each section, which contains the results for one question
    for section in sections:
        # Split the section into lines for easier processing
        lines = section.strip().split('\n')
        
        result = []  # Temporary list to hold the response for the current question
        source_list = []  # Temporary list to hold sources for the current question
        
        in_result_section = False  # Flag to track whether we are reading the result section of the current question

        # Process each line to extract the response and sources
        for line in lines:
            if line.startswith("Result:"):  # If the line starts with "Result:", it's part of the response
                in_result_section = True  # We are now in the result section
                result.append(line[len("Result:"):].strip())  # Extract the response and strip unnecessary spaces
            elif line.startswith("Source:"):  # If the line starts with "Source:", it marks the end of the result section
                in_result_section = False  # We leave the result section
            elif line.startswith("Kilde:"):  # If the line starts with "Kilde:", it's a source line
                # Extract and store the source, removing unnecessary spaces
                source_list.append(line[len("Kilde:"):].strip())
            elif in_result_section:  # If we are still inside the result section, append further lines to the response
                result.append(line.strip())
        
        # Join all response lines into a single string and store them in the responses list
        responses.append(" ".join(result))
        # Append the list of sources for this question to the sources list
        sources.append(source_list)
    
    #################################################################
    # Initialize variables for testing analysis
    #################################################################

    # Initialize variables to track the total number of positive and negative tests
    total_antall_negative_tester = 0  # Total number of negative tests
    total_antall_positive_tester = 0  # Total number of positive tests
    
    # Initialize success-related counters
    total_suksessfulle_tester = 0  # Total number of successful tests
    total_suksessfulle_positive_tester = 0  # Number of successful positive tests
    total_suksessfulle_negative_tester = 0  # Number of successful negative tests
    
    # Initialize failure-related counters
    total_feil_tester = 0  # Total number of failed tests
    total_feil_positive_tester = 0  # Number of failed positive tests
    total_feil_negative_tester = 0  # Number of failed negative tests
    
    #################################################################
    # Initialize variables for positive tests analysis
    #################################################################
    # Track correct and incorrect matches for positive tests
    positiv_samsvar_riktigkilde = 0  # Correct match with the right source
    positiv_samsvar_feilkilde = 0  # Incorrect match with the wrong source
    positiv_ikkesamsvar_riktigkilde = 0  # Correct answer but mismatch with the right source
    positiv_ikkesamsvar_feilkilde = 0  # Incorrect answer and mismatch with the wrong source

    #################################################################
    # Initialize variables for negative tests analysis
    #################################################################
    # Track correct and incorrect matches for negative tests
    negativ_samsvar_riktigkilde = 0  # Incorrect match with the right source
    negativ_samsvar_feilkilde = 0  # Incorrect match with the wrong source
    negativ_ikkesamsvar_riktigkilde = 0  # Correct answer but mismatch with the right source
    negativ_ikkesamsvar_feilkilde = 0  # Correct answer and mismatch with the wrong source

    #################################################################
    # Initialize uncertainty tracking variables
    #################################################################
    # Track uncertainty-related statistics
    usikker_count = 0  # Total uncertain results
    usikker_positiv_riktigkilde_count = 0  # Uncertain positive with the right source
    usikker_positiv_feilkilde_count = 0  # Uncertain positive with the wrong source
    usikker_negativ_riktigkilde_count = 0  # Uncertain negative with the right source
    usikker_negativ_feilkilde_count = 0  # Uncertain negative with the wrong source

    #################################################################
    # Initialize list for collecting all test failures
    #################################################################
    failures = []  # List to store all failures for analysis

    #################################################################
    # Start the LLM (Language Model) for further processing
    #################################################################
    mistral_llm = setupLLM()  # Initialize the LLM using the previously defined setupLLM function

    #################################################################
    # Variables for expected responses, sources, and negative cases
    #################################################################
    expected_responses = []  # List for expected responses from the model
    expected_sources = []  # List for expected sources for each question
    negative_cases = []  # List for negative test cases to analyze incorrect responses

    # Open the expected_responses_sources.txt file for reading and process the data
    with open("expected_responses_sources.txt", 'r', encoding='utf-8', errors='replace') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Iterate over each line to extract expected responses, sources, and negative cases
    for line in lines:
        # Check if the line contains an expected response or source
        if line.startswith("Expected Response:"):
            # Extract the response and strip any unnecessary whitespace around it
            response = line[len("Expected Response:"):].strip()
            expected_responses.append(response)  # Store the extracted response in the expected_responses list
        elif line.startswith("Expected Source:"):
            # Extract the source and strip any unnecessary whitespace
            source = line[len("Expected Source:"):].strip()
            expected_sources.append(source)  # Store the extracted source in the expected_sources list
        elif line.startswith("Negative Case:"):
            # Extract the negative case (if any) and ensure it's in lowercase for uniformity
            case = line[len("Negative Case:"):].strip().lower()
            negative_cases.append(case)  # Store the negative case in the negative_cases list
    
    #################################################################
    # Validate test cases by comparing responses with expected outcomes
    #################################################################

    # Loop through each expected response and compare it with the actual response from the system
    for test_count, expected_response in enumerate(expected_responses):
        # Get the corresponding negative case, actual response, and sources for the current test case
        negative_case = negative_cases[test_count]
        actual_response = responses[test_count].strip()
        actual_sources = sources[test_count]
        expected_source = expected_sources[test_count]
        
        #################################################################
        # Prepare the prompt for evaluation, formatted with expected and actual responses
        #################################################################

        # Create a formatted prompt for evaluation purposes, to assess if the model's response matches the expected response
        prompt = EVAL_PROMPT.format(
            expected_response=expected_response,
            actual_response=actual_response
        )

        #################################################################
        # Set up the evaluation chain to check for correctness of the model's response
        #################################################################

        # Create an evaluation chain using the Language Model (LLM) to compare expected and actual responses
        eval_chain = LLMChain(
            llm=mistral_llm,  # Use the previously set-up LLM model
            prompt=PromptTemplate(input_variables=["expected_response", "actual_response"], template=EVAL_PROMPT)
        )

        #################################################################
        # Run the evaluation chain and handle any potential errors (like out-of-memory issues)
        #################################################################

        try:
            # Run the evaluation and get the result
            eval_result = eval_chain.run({
                "expected_response": expected_response,
                "actual_response": actual_response
            })
        except RuntimeError as e:
            # Handle runtime errors, especially memory-related issues (e.g., out of memory in CUDA)
            if 'CUDA out of memory' in str(e):
                print("Test failed due to memory issues, retrying...")
                time.sleep(10)  # Wait for a while before retrying the evaluation
                gc.collect()  # Trigger garbage collection to free up GPU/CPU memory
                continue  # Skip to the next iteration (retry the evaluation)

        #################################################################
        # Process the evaluation result for consistency and cleanliness
        #################################################################

        # Clean up the evaluation result: 
        # Remove unnecessary characters, ensure the response is lowercased, and strip whitespace
        eval_result_cleaned = eval_result.split("<|im_start|> assistant")[-1].strip()

        # Convert the evaluation result to lowercase for uniform comparison
        eval_result_cleaned = eval_result_cleaned.lower()


        # Debugging: Optionally print the cleaned evaluation result
        #print(f"Evaluation Result for Test {test_count+1}: '{eval_result_cleaned}' (Expected: '{expected_response}')")

        # Initialize a flag to track if the source matches
        source_match = False

        # Iterate over the actual sources and check if any of them match the expected source
        for source in actual_sources:
            # Use strip() to remove extra whitespace and lower() for case-insensitivity when comparing
            if expected_source.strip().lower() == source.strip().lower():
                source_match = True  # Source matches, so set the flag to True
                break  # Exit the loop early if a match is found

        # Prepare a formatted result string to write into the output file
        result_entry = f"Test {test_count+1}:\n"
        result_entry += f"Expected Response: {expected_response}\n"
        result_entry += f"Actual Response: {actual_response}\n"
        result_entry += f"Evaluation Result: {eval_result_cleaned}\n"

        ######################################################
        # Checking the result to see if it matches or not
        # Specifically, we are handling true/false cases in the evaluation
        ######################################################

        # Check for positive test cases (when the expected response should be "true")
        if negative_case == "false":
            total_antall_positive_tester += 1  # Increase count for positive tests
            if "true" in eval_result_cleaned or "sammenheng" in eval_result_cleaned or "det faktiske svaret samsvarer med det forventede svaret" in eval_result_cleaned or "samsvarer" in eval_result_cleaned or "sant" in eval_result_cleaned or "tilsvarer" in eval_result_cleaned or "samsvar" in eval_result_cleaned or "samsvare" in eval_result_cleaned or "det samme som" in eval_result_cleaned or "korresponderte" in eval_result_cleaned or "korresponderer" in eval_result_cleaned or "korresponderer faktisk" in eval_result_cleaned:
                # If the evaluation result indicates "true" and the source matches
                if source_match == True:
                    total_suksessfulle_positive_tester += 1  # Increment successful positive tests
                    total_suksessfulle_tester += 1  # Increment total successful tests
                    positiv_samsvar_riktigkilde += 1  # Increment the count for correct positive source matches
                    result_entry += "Status: Success (Both response and source are correct)\n"
                else:
                    # Source does not match, mark as failure
                    positiv_samsvar_feilkilde += 1  # Increment incorrect source count
                    total_feil_tester += 1  # Increment total failures
                    total_feil_positive_tester += 1  # Increment failures for positive tests
                    result_entry += "Status: Failed Positive test(Correct answer but wrong source)\n"
                    failures.append(f"Test {test_count+1} failed: Correct answer but wrong source")
            elif "false" in eval_result_cleaned or "ikke sammenheng" in eval_result_cleaned or "ikke samsvarer" in eval_result_cleaned or "usant" in eval_result_cleaned or "samsvarer ikke" in eval_result_cleaned or "skiller seg betydelig" in eval_result_cleaned or "stemmer ikke overens" in eval_result_cleaned or "ikke i samsvar" in eval_result_cleaned or "korresponderte ikke" in eval_result_cleaned or "korresponderer ikke" in eval_result_cleaned or "motsier" in eval_result_cleaned or "motsi" in eval_result_cleaned:
                # If the evaluation result indicates "false" and the source matches
                if source_match == True:
                    positiv_ikkesamsvar_riktigkilde += 1  # Increment for wrong answer but correct source
                    total_feil_tester += 1  # Increment total failures
                    total_feil_positive_tester += 1  # Increment failure count for positive tests
                    result_entry += "Status: Failed Positive Test(Wrong answer but correct source)\n"
                    failures.append(f"Test {test_count+1} failed: Wrong answer but correct source")
                else:
                    # Both the answer and source are incorrect
                    positiv_ikkesamsvar_feilkilde += 1  # Increment for wrong answer and wrong source
                    total_feil_tester += 1  # Increment total failures
                    total_feil_positive_tester += 1  # Increment failures for positive tests
                    result_entry += "Status: Failed Positive Test(Wrong answer and wrong source)\n"
                    failures.append(f"Test {test_count+1} failed: Wrong answer and wrong source")
            else:
                # If the result is uncertain (not clearly true or false), handle separately
                if source_match == True:
                    total_antall_positive_tester -= 1  # Decrease positive test count for uncertain results
                    usikker_count += 1  # Increase uncertainty count
                    usikker_positiv_riktigkilde_count += 1  # Increase count for uncertain positive responses with correct source
                    result_entry += "Status: Failed Positive Test(Uncertainty but correct source)\n"
                    failures.append(f"Test {test_count+1} failed(positive): Uncertain answer but correct source")
                else:
                    total_antall_positive_tester -= 1  # Decrease count for positive tests with uncertainty
                    usikker_count += 1  # Increase uncertainty count
                    usikker_positiv_feilkilde_count += 1  # Increase count for uncertain positive responses with wrong source
                    result_entry += "Status: Failed Positive Test(Uncertainty response with wrong source)\n"
                    failures.append(f"Test {test_count+1} failed Positive Test: Uncertainty in the model's response")

        ######################################################
        # Checking negative test cases (where the expected response should be "false")
        ######################################################

        if negative_case == "true":
            total_antall_negative_tester += 1  # Increase count for negative tests
            if "false" in eval_result_cleaned or "ikke sammenheng" in eval_result_cleaned or "ikke samsvarer" in eval_result_cleaned or "usant" in eval_result_cleaned or "samsvarer ikke" in eval_result_cleaned or "skiller seg betydelig" in eval_result_cleaned or "stemmer ikke overens" in eval_result_cleaned or "ikke i samsvar" in eval_result_cleaned or "korresponderte ikke" in eval_result_cleaned or "korresponderer ikke" in eval_result_cleaned or "motsier" in eval_result_cleaned or "motsi" in eval_result_cleaned:
                # If the evaluation result indicates "false" and the source matches
                if source_match == True:
                    negativ_ikkesamsvar_feilkilde += 1  # Increase for correct answer but wrong source in negative cases
                    total_feil_tester += 1  # Increase total failures
                    total_feil_negative_tester += 1  # Increase failures for negative tests
                    result_entry += "Status: Failed Negative Test(Correct answer but wrong negative case source)\n"
                    failures.append(f"Test {test_count+1} failed: Negative Test Correct answer but wrong source")
                else:
                    # If both answer and source are incorrect for a negative test case
                    negativ_ikkesamsvar_riktigkilde += 1  # Increment for incorrect answer but correct source
                    total_suksessfulle_negative_tester += 1  # Increment successful negative tests
                    total_suksessfulle_tester += 1  # Increment total successful tests
                    result_entry += "Status: Success Negative Case (Both response and source are correct)\n"
            elif "true" in eval_result_cleaned or "sammenheng" in eval_result_cleaned or "det faktiske svaret samsvarer med det forventede svaret" in eval_result_cleaned or "samsvarer" in eval_result_cleaned or "sant" in eval_result_cleaned or "tilsvarer" in eval_result_cleaned or "samsvar" in eval_result_cleaned or "samsvare" in eval_result_cleaned or "det samme som" in eval_result_cleaned or "korresponderte" in eval_result_cleaned or "korresponderer" in eval_result_cleaned or "korresponderer faktisk" in eval_result_cleaned:
                # If the evaluation result indicates "true" but source does not match
                if source_match == True:
                    negativ_samsvar_feilkilde += 1  # Incorrect answer and incorrect source in negative case
                    total_feil_tester += 1  # Increment failures
                    total_feil_negative_tester += 1  # Increment failures for negative tests
                    result_entry += "Status: Failed Negative Test(Wrong answer and wrong negative case source)\n"
                    failures.append(f"Test {test_count+1} failed: Wrong answer and wrong source")
                else:
                    # Wrong answer but correct source
                    negativ_samsvar_riktigkilde += 1  # Increment for wrong answer but correct source
                    total_feil_tester += 1  # Increment total failures
                    total_feil_negative_tester += 1  # Increment failures for negative tests
                    result_entry += "Status: Failed Negative Test(Wrong answer but correct negative-case source)\n"
                    failures.append(f"Test {test_count+1} failed: Wrong answer but correct source")
            else:
                # If the evaluation is uncertain for negative test cases
                if source_match == True:
                    total_antall_negative_tester -= 1  # Decrease negative test count for uncertainty
                    usikker_negativ_riktigkilde_count += 1  # Increment uncertain negative responses with correct source
                    usikker_count += 1  # Increase uncertainty count
                    result_entry += "Status: Failed Negative Test(Uncertainty and wrong source)\n"
                    failures.append(f"Test {test_count+1} failed: Uncertain answer but wrong source")
                else:
                    total_antall_negative_tester -= 1  # Decrease negative test count for uncertainty
                    usikker_count += 1  # Increase uncertainty count
                    usikker_negativ_feilkilde_count += 1  # Increase uncertain negative responses with wrong source
                    result_entry += "Status: Failed Negative Test(Uncertainty in the model's response with correct source)\n"
                    failures.append(f"Test {test_count+1} failed: NEGATIVE CASE Uncertainty in the model's response")
                    
    # After evaluating, append the result to the output file
    with open("LLM_Matching_Result.txt", "a", encoding='utf-8', errors='replace') as llm_file:
        # Append the results of the current test to the file
        llm_file.write(result_entry + "\n")

    ##############################################################
    # Summarizing the results of all tests (including uncertain cases)
    ##############################################################
    
    # Calculate total number of tests including those with uncertainty
    total_tester = total_antall_negative_tester + total_antall_positive_tester + usikker_count
    
    # Total number of tests excluding uncertain cases
    total_tester_utenusikkerhet = total_antall_negative_tester + total_antall_positive_tester
    
    # Total tests with uncertainty in the negative cases
    total_antall_tester_medusikkerhet = total_antall_negative_tester + usikker_negativ_feilkilde_count + usikker_negativ_riktigkilde_count
    
    # Total number of tests without uncertainty for negative tests
    total_antall_tester_utenusikkerhet = total_antall_negative_tester
    
    # Total number of positive tests, including those with uncertain results
    total_antall_positive_tester_medusikkerhet = total_antall_positive_tester + usikker_positiv_feilkilde_count + usikker_positiv_riktigkilde_count
    
    # Total number of positive tests excluding those with uncertainty
    total_antall_tester_positive_utenusikkerhet = total_antall_positive_tester
    
    ##############################################################
    # Calculating success rates in decimal form
    ##############################################################
    
    # Calculate the decimal success rate for positive tests without uncertainty
    total_desimal_suksessfulle_tester = total_suksessfulle_positive_tester / total_antall_tester_positive_utenusikkerhet
    
    # Calculate the decimal success rate for positive tests including uncertainty
    total_desimal_suksessfulle_tester_med_usikkerhet = total_suksessfulle_positive_tester / total_antall_positive_tester_medusikkerhet
    
    # Calculate the decimal success rate for negative tests without uncertainty
    total_desimal_negative_tester = total_suksessfulle_negative_tester / total_antall_tester_utenusikkerhet
    
    # Calculate the decimal success rate for negative tests including uncertainty
    total_desimal_negative_tester_med_usikkerhet = total_suksessfulle_negative_tester / total_antall_tester_medusikkerhet
    
    ##############################################################
    # Calculating the average success rate for all tests
    ##############################################################
    
    # Calculate the average success rate across all tests, including uncertain ones
    gjennomsnittlig_rate_med_usikkerhet = total_suksessfulle_tester / total_tester
    
    # Calculate the average success rate across all tests, excluding uncertain ones
    gjennomsnittlig_rate_uten_usikkerhet = total_suksessfulle_tester / total_tester_utenusikkerhet
    
    ##############################################################
    # Prepare a summary entry that will be written to the file
    ##############################################################
    
    summary_entry = ""
    summary_entry += f"\n===== Test Summary =====\n"
    summary_entry += f"Total tests run (with uncertainty): {total_tester}\n"
    summary_entry += f"Total tests run (without uncertainty): {total_tester_utenusikkerhet}\n"
    summary_entry += f"Total Positive tests run: {total_antall_positive_tester}\n"
    summary_entry += f"Total Negative tests run: {total_antall_negative_tester}\n"
    summary_entry += f"Average success rate (with uncertainty): {gjennomsnittlig_rate_med_usikkerhet}\n"
    summary_entry += f"Average success rate (without uncertainty): {gjennomsnittlig_rate_uten_usikkerhet}\n"
    
    summary_entry += f"\n===== Successful Tests =====\n"
    summary_entry += f"Total number of successful tests: {total_suksessfulle_tester}\n"
    summary_entry += f"Total number of successful Positive-Case tests: {total_suksessfulle_positive_tester}\n"
    summary_entry += f"Total number of successful Negative-Case tests: {total_suksessfulle_negative_tester}\n"
    summary_entry += f"Decimal of Successful Positive-Case tests (with uncertain cases): {total_desimal_suksessfulle_tester_med_usikkerhet}\n"
    summary_entry += f"Decimal of Successful Positive-Case tests (without uncertain cases): {total_desimal_suksessfulle_tester}\n"
    summary_entry += f"Decimal of Successful Negative-Case tests (with uncertain cases): {total_desimal_negative_tester_med_usikkerhet}\n"
    summary_entry += f"Decimal of Successful Negative-Case tests (without uncertain cases): {total_desimal_negative_tester}\n"
    
    summary_entry += f"\n===== Failed Tests =====\n"
    summary_entry += f"Total number of failed tests: {total_feil_tester}\n"
    summary_entry += f"Total number of failed Positive-Case tests: {total_feil_positive_tester}\n"
    summary_entry += f"Total number of failed Negative-Case tests: {total_feil_negative_tester}\n"
    
    summary_entry += "\n===== Info about failed Positive-Case Tests =====\n"
    summary_entry += f"Tests with Matching RAG-Response and wrong source: {positiv_samsvar_feilkilde}\n"
    summary_entry += f"Tests with Wrong RAG-Response and correct source: {positiv_ikkesamsvar_riktigkilde}\n"
    summary_entry += f"Tests with Wrong RAG-Response and wrong source: {positiv_ikkesamsvar_feilkilde}\n"
    
    summary_entry += f"\n===== Info about failed Negative-Case Tests =====\n"
    summary_entry += f"Tests with Correct RAG-Response and wrong source: {negativ_ikkesamsvar_feilkilde}\n"
    summary_entry += f"Tests with Wrong RAG-response and correct source: {negativ_samsvar_riktigkilde}\n"
    summary_entry += f"Tests with wrong RAG-Response and wrong source: {negativ_samsvar_feilkilde}\n"
    
    summary_entry += f"\n===== Info about Uncertain Tests =====\n"
    summary_entry += f"Uncertain results: {usikker_count}\n"
    summary_entry += f"Uncertain Positive Case results with correct source: {usikker_positiv_riktigkilde_count}\n"
    summary_entry += f"Uncertain Positive Case results with wrong source: {usikker_positiv_feilkilde_count}\n"
    summary_entry += f"Uncertain Negative Case results with correct source: {usikker_negativ_feilkilde_count}\n"
    summary_entry += f"Uncertain Negative Case results with wrong source: {usikker_negativ_riktigkilde_count}\n"
    
    summary_entry += f"\n===== Summary about all failures =====\n"
    summary_entry += f"All the failures: {failures}\n"
    
    ##############################################################
    # Write the summary entry to the output file
    ##############################################################
    with open("LLM_Matching_Result.txt", "a", encoding='utf-8', errors='replace') as llm_file:
        llm_file.write(summary_entry)
        
    
# Ensure the main function is being called correctly
if __name__ == "__main__":
    fase2()
