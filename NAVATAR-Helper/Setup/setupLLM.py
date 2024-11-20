import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
import nest_asyncio


def setupLLM():
    #################################################################
    # Tokenizer
    #################################################################

    ## I denne variabelen setter man LLM-modellen.
    ## For eksempel, om man vil bruke nb-gpt, kan man sette model_name til 'NbAiLab/nb-gpt-j-6B'.
    model_name='norallm/normistral-7b-warm-instruct'

    # Laster ned konfigurasjonen for den spesifikke modellen
    model_config = transformers.AutoConfig.from_pretrained(model_name)

    # Laster inn tokenizeren, som konverterer tekst til tokens som modellen forstår
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Setter padding- og slutt-token til samme verdi for å sikre riktig justering
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ## Setter opp en chat-template for norallm/normistral-7b-warm-instruct.
    ## Hvis man bruker en annen modell, kan det være nødvendig å endre dette formatet.
    messages = [
        {"role": "user", "content": "Hva er hovedstaden i Norge?"},
        {"role": "assistant", "content": "Hovedstaden i Norge er Oslo. Denne byen ligger i den sørøstlige delen av landet, ved Oslofjorden. Oslo er en av de raskest voksende byene i Europa, og den er kjent for sin rike historie, kultur og moderne arkitektur. Noen populære turistattraksjoner i Oslo inkluderer Vigelandsparken, som viser mer enn 200 skulpturer laget av den berømte norske skulptøren Gustav Vigeland, og det kongelige slott, som er den offisielle residensen til Norges kongefamilie. Oslo er også hjemsted for mange museer, gallerier og teatre, samt mange restauranter og barer som tilbyr et bredt utvalg av kulinariske og kulturelle opplevelser."},
        {"role": "user", "content": "Gi meg en liste over de beste stedene å besøke i hovedstaden"}
    ]
    # Bruker tokenizeren til å konvertere meldingene til tokens, og legger til et genereringsprompt for modellen
    gen_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    #################################################################
    # bitsandbytes parameters
    #################################################################

    # Angir om modellen skal lastes inn med 4-bit presisjon (som sparer minne)
    use_4bit = True

    # Angir datatype som brukes for beregninger med 4-bit-modeller
    bnb_4bit_compute_dtype = "float16"

    # Velger kvantiseringstype (fp4 eller nf4). Kvantisering reduserer modellens størrelse uten å tape for mye nøyaktighet.
    bnb_4bit_quant_type = "nf4"

    # Angir om dobbelt kvantisering skal brukes (gir ytterligere komprimering, men kan påvirke ytelsen)
    use_nested_quant = False

    #################################################################
    # Set up quantization config
    #################################################################
    # Setter opp presisjonstypen for beregninger basert på valgt datatype (her float16)
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Oppretter konfigurasjonen for BitsAndBytes. Denne delen optimaliserer modellens ytelse ved å bruke 4-bit kvantisering. 
    # Dette betyr at modellen bruker mindre ressurser (som minne), noe som er spesielt nyttig for store språkmodeller.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Sjekker om GPU-en som brukes, støtter bfloat16 presisjon for bedre ytelse
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    #################################################################
    # Load pre-trained config
    #################################################################
    # Laster inn den forhåndstrente modellen med kvantisering (BitsAndBytesConfig) for å optimalisere minnebruk
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    # Skriver ut hvor mange av modellens parametere som kan trenes
    print(print_number_of_trainable_model_parameters(model))

    ## Disse parametrene styrer tekstgenereringen. Hvis du bytter modell, må disse justeres.
    ## Verdiene her er optimalisert for normistral instruct-modellen (norallm/normistral-7b-warm-instruct).
    text_generation_pipeline = pipeline(
        model=model,
        task="text-generation",  # Angir at vi bruker modellen for tekstgenerering
        tokenizer=tokenizer,
        max_new_tokens=1024,  # Hvor mange tokens modellen skal generere i ett output
        top_k=64,  # Begrens hvor mange alternativer modellen vurderer ved hvert steg
        top_p=0.9,  # Nucleus-sampling: velg blant de mest sannsynlige alternativene som dekker 90% sannsynlighet
        temperature=0.1,  # Lav temperatur gir mer konsistente (mindre tilfeldige) resultater
        repetition_penalty=1.0,  # Ingen straff for repetisjon (men kan endres om ønsket)
        do_sample=True,  # Tillat sampling for å få mer varierte svar
        use_cache=True  # Bruk cache for raskere generering
    )

    # Pakker genereringspipen inn i HuggingFacePipeline for enklere integrering med andre verktøy
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    ## Dette gjør at asynkrone funksjoner kan brukes sammen med andre ikke-asynkrone funksjoner.
    import nest_asyncio
    nest_asyncio.apply()

    return mistral_llm


def print_number_of_trainable_model_parameters(model):
    # Teller hvor mange parametere som kan trenes (requires_grad = True) og hvor mange totale parametere det er.
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    # Returnerer informasjon om hvor mange prosent av modellens parametere som kan trenes
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
