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

#Variables for time
tid = 0
antallspm = 0
gjennomsnittligtid = 0

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

    # Define the model name (pre-trained Norwegian model)
    model_name = 'norallm/normistral-7b-warm-instruct'

    # Load the model configuration from HuggingFace
    model_config = transformers.AutoConfig.from_pretrained(model_name)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to match the EOS token
    tokenizer.padding_side = "right"          # Pad sequences on the right side

    # Example chat messages for setting up a chat-based interaction
    messages = [
        {"role": "user", "content": "Hva er hovedstaden i Norge?"},
        {"role": "assistant", "content": "Hovedstaden i Norge er Oslo. Denne byen ligger i den sørøstlige delen av landet, ved Oslofjorden. Oslo er en av de raskest voksende byene i Europa, og den er kjent for sin rike historie, kultur og moderne arkitektur. Noen populære turistattraksjoner i Oslo inkluderer Vigelandsparken, som viser mer enn 200 skulpturer laget av den berømte norske skulptøren Gustav Vigeland, og det kongelige slott, som er den offisielle residensen til Norges kongefamilie. Oslo er også hjemsted for mange museer, gallerier og teatre, samt mange restauranter og barer som tilbyr et bredt utvalg av kulinariske og kulturelle opplevelser."},
        {"role": "user", "content": "Gi meg en liste over de beste stedene å besøke i hovedstaden"}
    ]

    # Tokenize the example messages and prepare them for the model
    gen_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    #################################################################
    # bitsandbytes parameters
    #################################################################

    # Enable 4-bit precision for model loading to optimize memory and performance
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"  # Specify computation data type (float16)
    bnb_4bit_quant_type = "nf4"         # Use normalized 4-bit floating-point quantization
    use_nested_quant = False            # Disable nested quantization for simplicity

    #################################################################
    # Set up quantization config
    #################################################################
    
    # Convert the string dtype into the corresponding torch type
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Define the Bits and Bytes (bnb) configuration for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant
    )

    # Check if the GPU supports bfloat16 for optimization purposes
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()  # Get GPU capability
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    #################################################################
    # Load pre-trained config
    #################################################################

    # Load the pre-trained language model with the defined quantization settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )

    #################################################################
    # Text generation pipeline
    #################################################################

    # Configure the text generation pipeline with parameters recommended by the model provider
    text_generation_pipeline = pipeline(
        model=model,
        task="text-generation",
        tokenizer=tokenizer,
        max_new_tokens=1024,      # Maximum number of tokens in generated text
        top_k=64,                 # Top-k sampling for diversity in generation
        top_p=0.9,                # Nucleus sampling
        temperature=0.1,          # Low temperature for more deterministic outputs
        repetition_penalty=1.0,   # No penalty for repetition
        do_sample=True,           # Enable sampling for generation
        use_cache=True            # Use caching for faster response times
    )

    # Wrap the pipeline with HuggingFace's interface for language models
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Apply nested asyncio to handle asynchronous calls
    nest_asyncio.apply()

    # Return the configured language model
    return mistral_llm

def createRagChain(mistral_llm):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain for answering questions in Norwegian about NEET,
    using a specific language model (Mistral) and a Milvus vector database.

    Args:
        mistral_llm: The language model (Mistral) used for generating responses.

    Returns:
        A RAG chain that retrieves relevant context from the database and generates responses.
    """

    # Path to the Milvus database
    DATABASE_PATH = "./../Database/Testing.db"

    # Initialize the Milvus database retriever with a multilingual embedding model
    db = Milvus(
        embedding_function=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct'),
        connection_args={"uri": DATABASE_PATH}
    )
    
    # Define the retriever to use similarity search with top-k results (k=6)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Prompt template for generating responses
    prompt_template = """
    <|im_start|> user
    Instruksjon:
    Du er en assistent som skal svare på spørsmål. Svar på norsk på spørsmålet basert på din kunnskap om NEET. Her er kontekst som kan hjelpe, bruk kun kunnskap fra dette til å svare på spørsmålet:

    {kontekst}

    Spørsmål:
    {spørsmål}<|im_end|>
    <|im_start|> assistant
    """

    # Define the prompt template with placeholders for context and question
    prompt = PromptTemplate(
        input_variables=["kontekst", "spørsmål"],  # Input variables for the template
        template=prompt_template,                # Template string with structure
    )
    
    # Create an LLM chain combining the language model with the prompt template
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
    
    # Construct the RAG chain:
    # 1. Use the retriever to fetch relevant context
    # 2. Pass the raw question through RunnablePassthrough (identity transformation)
    # 3. Combine both inputs to generate a response using the LLM chain
    rag_chain = {
        "kontekst": retriever,               # Retrieves context based on the query
        "spørsmål": RunnablePassthrough()   # Passes the question directly without changes
    } | llm_chain                           # Combines retrieved context and the question in the LLM chain

    return rag_chain  # Return the constructed RAG chain


################################################################################################################################################
####################################### TEST CASES #############################################################################################
################################################################################################################################################

def ungdomsdepresjon():
    question="Hvordan er ungdomsdepresjon relatert til arbeidsledighet i voksen alder?"
    expected_response="Ungdomsdepresjon er funnet å være assosiert med arbeidsledighet i voksen alder"
    expected_source="AdolescentDepressionPhysosocialOutcomes.pdf"
    negativecase = False 
    return question,expected_response,expected_source,negativecase 
  
def TestresultaterNEET():
    question=" Hvordan kan gode testresultater påvirke effekten av fravær på arbeidsmarkedsutfallene?"
    expected_response = "Gode resultater i tester kan redusere effekten av fravær på arbeidsmarkedsutfallene. Dette betyr at fravær ikke nødvendigvis er et betydelig problem for unge mennesker i form av deres etter-skole utfall, så lenge denne oppførselen ikke reduserer testresultatene."
    expected_source="EffectsTestScoresTruancyNEET.pdf"
    negativecase= False
    return question,expected_response,expected_source,negativecase

def metaanalyse():
    question="Hva foreslår meta-analyser om effekten av dårlig helse i ungdomsårene på fremtidige resultater?"
    expected_response="Meta-analyser foreslo at dårlig helse i ungdomsårene generelt fører til dårligere utdannings- og arbeidsresultater i voksen alder."
    expected_source="AdolescentHealthAdultEducationEmployment.pdf"
    negativecase = False 
    return question,expected_response,expected_source,negativecase    

def reduksjonSannsynlighet():
    question="Hvilke faktorer reduserte sannsynligheten for å være i NEET-status?"
    expected_response="Positive holdninger til skolen, Det å ikke bli mobbet"
    expected_source="AdolescentPsychosocialFactorsMentalHealth.pdf"
    negativecase = False 
    return question,expected_response,expected_source,negativecase 
    
def ACEs():
    question="Hva er forholdet mellom negative barndomserfaringer (ACEs) og tung røyking i tidlig voksen alder, spesielt i landet ledet av Lai Ching-te?"
    expected_response="I Taiwan var ACEs i tidlig ungdomsalder betydelig relatert til tung røyking i tidlig voksen alder."
    expected_source="AdverseChildhoodHeavySmokingNEET.pdf"
    negativecase = False         
    return question,expected_response,expected_source,negativecase 

def Alkoholkonsum():
    question="Hvordan påvirker alkoholkonsum sannsynligheten for at elever får problemer på skolen, spesielt blant mennesker som kan føde barn?"
    expected_response="Kvinner som drikker alkohol har 11 % høyere sannsynlighet for å få problemer på skolen, noe som tyder på at alkoholkonsum har negative effekter, selv om disse effektene ikke nødvendigvis resulterer i dårligere karakterer."
    expected_source="AlcoholUseAcademicAchievementHS.pdf"
    negativecase = False       
    return question,expected_response,expected_source,negativecase 
    
def Chile():
    question="Hva er anbefalingen for fremtidige analyser og intervensjonsinnsatser rettet mot NEET-gruppen i Chile?"
    expected_response="Ifølge en studie fra 2022 er anbefalingen å bruke en kjønnsinformert tilnærming i analysen og intervensjonsinnsatsene rettet mot NEET-gruppen i Chile."
    expected_source="BarriersStudyingWorkingNEETChile.pdf"
    negativecase = False         
    return question,expected_response,expected_source,negativecase 

def Liberty():
    question="Liberty Bell er et historisk symbol på amerikansk uavhengighet lokalisert i Philadelphia, men hvordan har covid-19-pandemien påvirket barnas sosiale aktiviteter?"
    expected_response="Selv om Liberty Bell er et viktig symbol i Philadelphia, har pandemien ført til stenging av parker og lekeplasser samt begrensning av playdates, noe som betydelig har påvirket barns vanlige sosiale aktiviteter."
    expected_source="COVIDTechChildWellbeing.pdf"
    negativecase = False         
    return question,expected_response,expected_source,negativecase 

def Singapore():
    question="hvordan karakteriserer studier risikoen for NEET/Hikikomori-tendenser i Singapore?"
    expected_response=" Ifølge en studie fra 2021 kjennetegnes risikoen for NEET/Hikikomori-tendenser i Singapore av kulturell marginalisering, opplevd sosial avvisning og lavt selvbilde."
    expected_source="CultureHikikomoriSingapore.pdf"
    negativecase = False         
    return question,expected_response,expected_source,negativecase 

def Sorafrika():
    question="Hvilke faktorer ble funnet å øke sannsynligheten for ungdomsdeltakelse i jordbruksaktiviteter i Sør-Afrika?"
    expected_response="Studien fant at alder, statlig finansiering og foreldres deltakelse i landbruk øker sannsynligheten for unges deltakelse i jordbruksaktiviteter i Sør-Afrika."
    expected_source="DeterminantsRuralYouthAgricultureCapeSouthAfrica.pdf"
    negativecase = False        
    return question,expected_response,expected_source,negativecase

def SpaniaOvergang():
    question="Hvordan sammenligner skole-til-arbeid overgangssystemet i Spania seg med andre europeiske land?"
    expected_response="Spanias skole-til-arbeid overgangssystem er preget av høy ungdomsarbeidsløshet og svake koblinger mellom utdanning og arbeid. Overgangsprosessen er ofte heterogen, ikke-lineær, og uforutsigbar, og skiller seg fra andre europeiske land"
    expected_source="YouthUnemploymentSpain.pdf"     
    negativecase = False    
    return question,expected_response,expected_source, negativecase

def Personlighetstrekk():
    question="Hva er påvirkningen av lav selvtillit, lav innsats og flid, og en ekstern locus of control på sannsynligheten for å være ikke i utdanning, sysselsetting eller opplæring (NEET)?"
    expected_response="Individer som viser lav selvtillit, lav innsats og flid, og en ekstern locus of control står overfor en betydelig høyere sannsynlighet for å være NEET. Kombinasjonen av lav innsats og lav selvtillit ser ut til å være spesielt skadelig, og øker sjansene for å være NEET og forbli i denne tilstanden over lang tid."
    expected_source="YouthUnemploymentPersonalityTraits.pdf"   
    negativecase = False      
    return question,expected_response,expected_source, negativecase

def MENA():
    question="Hva er statusen for ungdomsarbeidsløshet i Midtøsten og Nord-Afrika (MENA) sammenlignet med resten av verden?"
    expected_response="Ungdomsarbeidsløshet i MENA-regionen er den høyeste i verden, og overgår alle andre geopolitiske regioner."
    expected_source="YouthUnemploymentArabSpring.pdf"  
    negativecase = False      
    return question,expected_response,expected_source,negativecase

def ArabiskVaar():
    question="Hvordan påvirket den arabiske våren, som startet i 2011, arbeidsmarkedet i MENA-regionen?"
    expected_response="Den arabiske våren resulterte i omfattende endringer som negativt påvirket arbeidsmarkedet i MENA-regionen, noe som førte til en betydelig økning i arbeidsløshetsgraden."
    expected_source="YouthUnemploymentArabSpring.pdf"  
    negativecase = False      
    return question,expected_response,expected_source,negativecase

def MexicoUtfordringer():
    question="Hvilke utfordringer står unge mennesker i Mexico overfor når det gjelder utdanning og sysselsetting?"
    expected_response=" Unge individer i Mexico står ofte overfor begrensede utdannings- og arbeidsmuligheter, med økende antall som ikke er i utdanning, sysselsetting eller opplæring (NEET). Hindringer for utdanning inkluderer lave opptakssatser til offentlige universiteter, økonomiske vanskeligheter, familieforpliktelser og vanskeligheter med å knytte skolegang og fremtidig sysselsetting. Arbeidsutfordringer inkluderer mangel på jobbmuligheter, diskriminering mot uerfarne arbeidere og uønskeligheten av lavtlønnede stillinger."
    expected_source="YouthNEETMexicoCityMixedMethodsAnalysis.pdf"  
    negativecase = False      
    return question,expected_response,expected_source,negativecase
#Negative test-cases
def HvilkePersonlighetstrekkFeil():
    question="Hvordan påvirker personlighetsegenskaper sannsynligheten for å forbli NEET over en utvidet periode?"
    expected_response="Personlighetsegenskaper har ingen påvirkning på sannsynligheten for å forbli NEET over en utvidet periode. Utdanningsbakgrunn og familieøkonomi er de eneste faktorene som har betydning."
    expected_source="NEETSwissYouthMentalDrugs.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def SpaniaOvergangFeil():
    question="Hvordan sammenligner skole-til-arbeid overgangssystemet i Spania seg med andre europeiske land?"
    expected_response="Skole-til-arbeid overgangssystemet i Spania er identisk med systemene i alle andre europeiske land, uten noen bemerkelsesverdige forskjeller i struktur, tilnærming eller resultater."
    expected_source="PsychosisDisconnectedYouthReview.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def AraniskVaarFeil():
    question="Hvordan påvirket den arabiske våren, som startet i 2011, arbeidsmarkedet i MENA-regionen?"
    expected_response="Den arabiske våren, som startet i 2011, hadde en positiv innvirkning på arbeidsmarkedet i MENA-regionen. Den førte til en betydelig nedgang i arbeidsløshetsratene og stimulerte økonomisk vekst og jobbskaping."
    expected_source="YoungNEETScotland.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def NEETBegrepFeil():
    question="Hva står NEET for og hvor stammer begrepet fra?"
    expected_response="NEET er en forkortelse for Norwegian Education and Employment Training og er et begrep som opprinnelig stammer fra Norge for å beskrive personer som er i utdanning, sysselsetting eller opplæring."
    expected_source="YouthLabourForceBarriers.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def NeetStatusMexicoCityFeil():
    question="Hvor utbredt er NEET-statusen blant unge mennesker i Mexico City?"
    expected_response="NEET-statusen er svært sjelden blant unge mennesker i Mexico City. Størstedelen av ungdommen er enten i utdanning, sysselsetting eller opplæring, og bare et marginalt antall er ikke det."
    expected_source="YoungItalianNEETsFamily.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase 
#Positive
def KompleksBehandling():
    question="Hvorfor bør behandling av komplekse problemer ikke overlates til psykiske helsetjenester alene?"
    expected_response="Behandling av komplekse problemer bør ikke overlates til psykiske helsetjenester alene fordi psykiske helseproblemer ofte maskerer større problemer. Oppfølging av sterke flerfaglige team, inkludert sosialarbeidere og helsepersonell, bør være blant tiltakene som iverksettes for NLFET-befolkningen."
    expected_source="YouthLabourForceBarriers.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def CovidItaly():
    question="Hva er de typiske hindringene for unge italienere som søker jobb etter endt utdanning?"
    expected_response="Tiden det tar for unge mennesker å finne en jobb etter endt utdanning i Italia er ekstremt lang. Før pandemien kunne det forventes at en person i aldersgruppen 20-24 år ville finne en permanent jobb etter 11,5 år eller en midlertidig jobb etter 5 år."
    expected_source="YoungPeopleCovidItaly.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def CovidItaly2():
    question="Hvordan har COVID-19-pandemien påvirket arbeidsmarkedet for unge mennesker i Italia?"
    expected_response="COVID-19-pandemien har forverret en allerede bekymringsfull situasjon med høyere inaktivitetsrater sammenlignet med andre EU-land. Det har blitt vanskeligere for unge mennesker å finne jobb, spesielt for kvinner og ikke-italienske borgere."
    expected_source="YoungPeopleCoviditaly.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase 

def BakgrunnAspirasjon():
    question="Hvordan kan familiens bakgrunn påvirke utviklingen av 'aspirasjonskapital' hos unge med minoritetsbakgrunn?"
    expected_response="Familiens bakgrunn kan betydelig påvirke utviklingen av 'aspirasjonskapital'. For eksempel kan foreldres forventninger og verdier sterkt innflytelse den unge personens ambisjoner og motivasjon for å lykkes."
    expected_source="YoungMinorityEthnicUKAspirationalCapital.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  
 
def ProblemerArbeid():
    question="Hvilke sosiale og helsemessige problemer møter unge voksne som er i fare for tidlig arbeidsuførhet?"
    expected_response="Denne gruppen møter betydelige utfordringer knyttet til negative sosiale opplevelser, psykologisk nød og alkoholbruk. De rapporterer ofte om mobbing, vold, og har høyere nivåer av psykologisk nød."
    expected_source="YoungAdultsEarlyWorkDisability.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def RusslandNEET():
    question="Hvordan påvirker høyere utdanning risikoen for å bli NEET i Russland?"
    expected_response="Høyere utdanning gir ikke en universell sikkerhetsnett fra NEET-status i Russland. Risiko for NEET-arbeidsledighet er assosiert med høyere utdanning på grunn av en overforsyning av kandidater med universitetsgrader."
    expected_source="WhatMakesYouthNEETEvidenceRussia.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def CovidGlobalt():
    question="Hvordan har COVID-19-krisen påvirket ungdoms arbeidsmarked globalt?"
    expected_response="COVID-19-krisen har alvorlig påvirket arbeidsmarkedene over hele verden, og har rammet unge mennesker mer enn andre aldersgrupper. Globalt falt ungdoms sysselsetting med 8,7 prosent i 2020, sammenlignet med 3,7 prosent for voksne."
    expected_source="UpdateYouthLabourImpactC19Crisis.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def DroppeUt():
    question="Hvordan påvirker tidspunktet for å droppe ut av universitetet risikoen for å bli NEET?"
    expected_response="Den negative effekten av tilbaketrekking på risikoen for å være NEET er langt verre for de studentene som tilbringer lengre tid på universitetet før de trekker seg ut. Dette er klart i samsvar med 'lock-in' hypotesen."
    expected_source="UniDOvsHighSchoolGD.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def Arbeidserfaring():
    question="Hvordan påvirker mangel på arbeidserfaring ungdoms sysselsettingsmuligheter?"
    expected_response="Mangelen på arbeidserfaring er den viktigste grunnen til at det er vanskelig for unge mennesker å finne en passende jobb. Arbeidsgivere er ofte motvillige til å ansette unge mennesker uten arbeidserfaring, inkludert universitetsutdannede."
    expected_source="TrendsYouthLaborMobility.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def IPS():
    question="Hvordan fungerer Individual Placement and Support (IPS) i forhold til tradisjonell yrkesrehabilitering for personer med alvorlig psykisk sykdom?"
    expected_response="Individual Placement and Support (IPS) er en yrkesrehabiliteringsprogram som er mer enn dobbelt så sannsynlig å føre til konkurransedyktig sysselsetting sammenlignet med tradisjonell yrkesrehabilitering for personer med alvorlig psykisk sykdom."
    expected_source="SupportedEmploymentMentalIllnessReview.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def Tjenestelering():
    question="Hva er hensikten med tjenestelæring i skolene?"
    expected_response="Tjenestelæring er en undervisningsstrategi som eksplisitt kobler samfunnstjeneste til akademisk instruksjon. Det er unikt fra tradisjonell frivillighet eller samfunnstjeneste fordi det bevisst knytter tjenesteaktiviteter med læreplanbegreper og inkluderer strukturert tid for refleksjon."
    expected_source="ServiceLearningAcademicSuccessGradeKTo12.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def KjennetegnSpania():
    question="Hva er noen av kjennetegnene for spanske NEETs?"
    expected_response="Spanske NEETs hadde generelt lavere utdanningsnivåer, var hovedsakelig arbeidsledige og gift, bortsett fra NEETs mellom 18 og 24 år, som heller var inaktive og single. De hadde også opplevd tidligere arbeidsledighet, hadde flere arbeidsledige venner, og kom fra fattigere familiebakgrunner sammenlignet med deres ikke-NEET motparter."
    expected_source="SchoolToWorkSpanishNEETs.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def FranceUtfordring():
    question="Hva er noen av de hovedutfordringene for NEETs i Frankrike?"
    expected_response="Mangel på utdanning og sosial kapital, samt geografiske økonomiske forhold, er avgjørende faktorer for å forbli i langvarige NEET-trajectorier."
    expected_source="SchoolToWorkFranceNEETsTrajectories.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def SenkeForventninger():
    question="Hva er noen av de potensielle negative effektene av å senke karriereforventninger og -mål?"
    expected_response="Selv om det å senke karriereforventninger og -mål kan øke sjansene for å finne jobb på kort sikt, kan det ha negative effekter på lang sikt. For eksempel kan det føre til at unge mennesker unngår videre utdanning og karriereutvikling, noe som kan begrense deres fremtidige inntektspotensial. Det kan også føre til lavere jobbtilfredshet og livstilfredshet."
    expected_source="ScarringDreamsYouthAspirationsUnemployment.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase  

def SorKoreaRisiko():
    question="Hva er noen av de viktigste risikofaktorene for å bli NEET i Sør-Korea?"
    expected_response="Fattigdom opplevd i ungdomstiden er en viktig risikofaktor for å bli NEET i tidlig voksen alder. Å ha en karriereplan og være fornøyd med skolelivet i løpet av ungdomsskolen og videregående skole reduserer risikoen for å bli NEET senere."
    expected_source="RiskFactorsNEETSouthKoreaPanelData.pdf"        
    negativecase=False
    return question,expected_response,expected_source,negativecase   
#Negative
def NLFET():
    question="Hva står NLFET for?"
    expected_response="NLFET står for Nordisk Lønns- og Finans Etterforskningsteam, som er en organisasjon som tar for seg økonomiske uregelmessigheter i de nordiske landene."
    expected_source="YoungPeopleCovidItaly"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def ItaliaCovidFeil():
    question="Hvordan har pandemien påvirket sannsynligheten for å finne midlertidig eller permanent arbeid i Italia?"
    expected_response="Pandemien har faktisk økt sannsynligheten for å finne midlertidig eller permanent arbeid i Italia, da mange bedrifter har utvidet og skapt flere jobbmuligheter for unge mennesker."
    expected_source="YoungNEETScotland.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase 

def AspirasjonFeil():
    question="Hva er rollen til sosial og kulturell kapital i utdannings- og karriereutviklingen til unge med minoritetsbakgrunn?"
    expected_response="Sosial og kulturell kapital har liten eller ingen rolle i utdannings- og karriereutviklingen til unge med minoritetsbakgrunn, da deres suksess hovedsakelig er avhengig av individuell innsats og talent."
    expected_source="YoungAdultsEarlyWorkDisability.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def ArsakerFeil():
    question="Hva er de selvoppfattede årsakene til sykdom blant unge voksne i fare for tidlig arbeidsuførhet?"
    expected_response="De selvoppfattede årsakene til sykdom blant unge voksne i fare for tidlig arbeidsuførhet er hovedsakelig relatert til dårlige kostholdsvaner og mangel på fysisk aktivitet."
    expected_source="SystematicOrScopingReview.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def RusslandFeil():
    question="Hva er de sosiodemografiske egenskapene til NEET-status (frafall fra sysselsetting, utdanning eller opplæring for unge mennesker mellom 15 og 24 år) i Russland?"
    expected_response="De sosiodemografiske egenskapene til NEET-status i Russland er hovedsakelig knyttet til unge mennesker fra høyinntektsfamilier med høy utdanning, siden de ofte velger å ta en pause fra utdanning eller arbeid for å reise eller engasjere seg i andre interesser."
    expected_source="TrendsYouthLaborMobility.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def CovidGjenomprettFeil():
    question="Hvordan har gjenopprettingen av ungdoms sysselsetting utviklet seg gjennom 2020?"
    expected_response="Gjenopprettingen av ungdoms sysselsetting har vært jevn og stabil gjennom 2020, med en kontinuerlig økning i sysselsettingen for unge mennesker i alle land, uavhengig av COVID-19-pandemiens innvirkning."
    expected_source="BeingNEETTurkeyDeterminantsConsequences.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def DroppUtFeil():
    question="Hvordan påvirker det å droppe ut av universitetet overgangen fra skole til arbeid?"
    expected_response="Å droppe ut av universitetet forbedrer overgangen fra skole til arbeid, da det gir unge mennesker tidlig praktisk erfaring og tidligere eksponering for arbeidsmarkedet, noe som øker deres sjanser for rask sysselsetting."
    expected_source="YouthLabourForceBarriers.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def ForbedreFeil():
    question="Hva anbefales for å forbedre situasjonen i ungdomsarbeidsmarkedet?"
    expected_response="For å forbedre situasjonen i ungdomsarbeidsmarkedet anbefales det å redusere fokus på yrkesutdanning og heller oppfordre unge mennesker til å gå rett inn i arbeidsstyrken etter videregående skole. Dette vil gi dem tidlig arbeidserfaring og hjelpe dem med å tilpasse seg arbeidsmiljøet raskere."
    expected_source="UnemploymentSubstanceAbuseUS2002.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def IPSFeil():
    question="Hvordan påvirker geografisk område, arbeidsledighetsrater eller veksten i bruttonasjonalproduktet effektiviteten til IPS?"
    expected_response="Geografisk område, arbeidsledighetsrater og veksten i bruttonasjonalproduktet har en betydelig innvirkning på effektiviteten av Individual Placement and Support (IPS). IPS er mest effektiv i områder med lav arbeidsledighet og høy økonomisk vekst, og mindre effektiv i områder med høy arbeidsledighet og lav økonomisk vekst."
    expected_source="SkillMatchingChallenge.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def SpaniaFeil():
    question="Hvordan har overgangen fra skole til arbeid endret seg i Spania?"
    expected_response="Overgangen fra skole til arbeid i Spania har blitt enklere og raskere, med de fleste unge mennesker som går direkte fra utdanning til faste fulltidsjobber."
    expected_source="SchoolToWorkFranceNEETsTrajectories.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def FranceFeil():
    question=" Hva er noen av de viktigste faktorene som kan forklare hvordan unge mennesker blir NEETs i Frankrike?"
    expected_response="En av de viktigste faktorene som kan forklare hvordan unge mennesker blir NEETs i Frankrike, er deres høye utdanningsnivå og rike familiebakgrunn."
    expected_source="RiskFactorsEducationOntarioCanada.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def aldersgruppeNEET():
    question="Hvilken aldersgruppe er mest sannsynlig å bli kategorisert som NEET?"
    expected_response="Eldre voksne over 65 år er mest sannsynlig å bli kategorisert som NEET."
    expected_source="ParentalSocioeconomicPredictorsNEETFinland.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  


def strategierFeil():
    question="Hva er noen strategier for å hjelpe NEETs tilbake til utdanning eller arbeid?"
    expected_response="En effektiv strategi for å hjelpe NEETs tilbake til utdanning eller arbeid er å gi dem fullstendig uavhengighet uten noen form for veiledning eller støtte. Ved å overlate dem helt til seg selv, vil de sannsynligvis finne motivasjonen og viljestyrken til å navigere i arbeidsmarkedet på egen hånd. Det vil også være nyttig å unngå å tilby dem ressurser som karriereveiledning, opplæringsprogrammer eller jobbsøkingsverktøy, da disse kan være overveldende og kanskje distrahere dem fra deres selvstendige søken. Videre er det viktig å ikke anerkjenne de psykologiske og økonomiske utfordringene de kan møte, da dette kan oppmuntre til en offermentalitet. I stedet bør man fremme en 'overlev de sterkeste' -tilnærming, hvor bare de mest motiverte og hardtarbeidende lykkes. På denne måten vil NEETs bli tvunget til å tilpasse seg og overkomme sine utfordringer på egen hånd, noe som til slutt kan føre til at de finner en jobb eller går tilbake til skolen."
    expected_source="NEETYouthMentalHealthServices.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def NeetStatistikkFeil():
    question="Hvilke land har høyest antall NEETs blant befolkningen og hvorfor?"
    expected_response="De landene som har det høyeste antallet NEETs blant befolkningen er de mest utviklede landene som Norge, Canada, Australia og Tyskland. Dette skyldes hovedsakelig deres høye levestandard og velutviklede velferdssystemer, som gir unge mennesker muligheten til å velge å være NEETs uten å oppleve alvorlige økonomiske konsekvenser. Disse landene har også robuste utdanningssystemer og arbeidsmarkeder, som paradoksalt nok kan virke avskrekkende på unge mennesker, ettersom de kan føle seg overveldet av de mange valgene og mulighetene de har. Videre kan det faktum at disse landene har høye inntektsnivåer også bidra til å øke antall NEETs, da unge mennesker kanskje ikke føler et press for å jobbe eller studere for å oppnå økonomisk stabilitet. Til slutt kan kulturelle faktorer også spille en rolle, da unge mennesker i disse landene kanskje verdsetter frihet og uavhengighet mer enn økonomisk sikkerhet eller akademiske prestasjoner. Derfor kan de velge å være NEETs for å utforske sine interesser, reise eller bare ta en pause fra det stressende livet."
    expected_source="InternalProblemsLikelihoodNEET.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  

def GenderFeil():
    question="Har kjønn noen innflytelse på sannsynligheten for å være NEET?"
    expected_response="Nei, kjønn har ingen innflytelse på sannsynligheten for å være NEET. Både menn og kvinner har samme risiko for å bli NEET, uavhengig av deres sosioøkonomiske bakgrunn, utdanningsnivå, eller personlige ambisjoner. Dette er fordi kjønn ikke påvirker en persons evne til å skaffe seg en jobb, lykkes i skolen, eller håndtere de ulike utfordringene som kan føre til at en person blir NEET. Faktisk har studier vist at kjønn har null betydning når det gjelder arbeidsledighet, utdanningsutfall, eller deltakelse i opplæring. Disse resultatene er konsistente på tvers av forskjellige kulturer og land, noe som indikerer at kjønn ikke har noen innvirkning på en persons NEET-status. På samme måte er det ingen kjønnsforskjeller i de negative effektene av å være NEET, som lavere livstilfredshet, dårligere psykisk helse, og økt risiko for fattigdom. Derfor er det klart at kjønn ikke har noen innflytelse på sannsynligheten for å være NEET."
    expected_source="InfluenceCrimeYoungNEET.pdf"        
    negativecase=True
    return question,expected_response,expected_source,negativecase  




##########################################################################################################################
####################################### TEST RUNNER ######################################################################
##########################################################################################################################

def run_all_tests():
    
    """
    Executes all predefined test functions, processes their outputs, and evaluates the 
    performance of a Retrieval-Augmented Generation (RAG) pipeline.

    The function performs the following:
        - Collects questions, expected responses, and sources from a list of test functions.
        - Writes the collected data to files for reference.
        - Configures a Language Model (LLM) and sets up a RAG pipeline for inference.
        - Invokes the RAG pipeline for each question, capturing response time and sources.
        - Logs results, including the average response time, to a results file.

    Returns:
        None
    """


    # Variables to track time
    global antallspm, tid, gjennomsnittligtid 

    # List of test functions
    test_functions = [
        ungdomsdepresjon,
        TestresultaterNEET,
        metaanalyse,
        reduksjonSannsynlighet,
        ACEs,
        Alkoholkonsum,
        Chile,
        Liberty,
        Singapore,
        Sorafrika,
        SpaniaOvergang,
        Personlighetstrekk,
        MENA,
        ArabiskVaar,
        MexicoUtfordringer,
        HvilkePersonlighetstrekkFeil,
        SpaniaOvergangFeil,
        AraniskVaarFeil,
        NEETBegrepFeil,
        NeetStatusMexicoCityFeil,
        KompleksBehandling,
        CovidItaly,
        CovidItaly2,
        BakgrunnAspirasjon,
        ProblemerArbeid,
        RusslandNEET,
        CovidGlobalt,
        DroppeUt,
        Arbeidserfaring,
        IPS,
        Tjenestelering,
        KjennetegnSpania,
        FranceUtfordring,
        SenkeForventninger,
        SorKoreaRisiko,
        NLFET,
        ItaliaCovidFeil,
        AspirasjonFeil,
        ArsakerFeil,
        RusslandFeil,
        CovidGjenomprettFeil,
        DroppUtFeil,
        ForbedreFeil,
        IPSFeil,
        SpaniaFeil,
        FranceFeil,
        aldersgruppeNEET,
        strategierFeil,
        NeetStatistikkFeil,
        GenderFeil
    ]
    
    questions = []
    expected_responses = []
    expected_sources = []
    negative_cases = []

    # Iterate over each test function and collect outputs
    for test_func in test_functions:
        # Call the test function to get its outputs
        question, expected_response, expected_source, negative_case = test_func()
        
        # Append the results to their respective lists
        questions.append(question)
        expected_responses.append(expected_response)
        expected_sources.append(expected_source)
        negative_cases.append(negative_case)

    # Open the file in append mode
    with open("expected_responses_sources.txt", "a", encoding='utf-8', errors='replace') as file:
        # Write expected responses and sources to the file
        for expected_response, expected_source, negative_case in zip(expected_responses, expected_sources, negative_cases):
            file.write(f"Expected Response: {expected_response}\n")
            file.write(f"Expected Source: {expected_source}\n")
            file.write(f"Negative Case: {negative_case}\n")
            file.write("\n")  # Newline for readability
            
    with open("questions.txt", "a", encoding='utf-8', errors='replace') as file:
        # Write the questions to a separate file
        for question in questions:
            file.write(f"Question: {question}")
            file.write("\n")  # Newline for readability

    # Setup models and RAG chain
    mistral_llm = setupLLM()
    rag_chain = createRagChain(mistral_llm)

    Count_question = 0
    for question in questions:
        
        # Start timing for this response
        antallspm += 1
        start_time = time.time()
        
        # Invoke the RAG chain to get the result
        result = rag_chain.invoke(question)
        
        # Stop timing for this response
        end_time = time.time()
        
        # Calculate the time taken for this response and add it to the total time
        response_time = end_time - start_time
        tid += response_time

        # Extract sources if available
        sources = [
            f"Source: {os.path.basename(doc.metadata['source'])}"
            for doc in result.get("kontekst", [])
        ]
        
        # Extract the response text
        response_text = result["text"].split("<|im_start|> assistant")[-1].strip()

        # Append the result and sources to the RAG_results.txt file
        with open("RAG_results.txt", "a", encoding='utf-8', errors='replace') as file:
            antallspm += 1  # Increment the question count
            Count_question += 1
            file.write(f"Question Number: {Count_question}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Result: {response_text}\n")
            
            # Append sources if found
            if sources:
                file.write("Source:\n")
                for source in sources:
                    file.write(f"{source}\n")

            file.write("\n")  # Newline for readability
            
    # Calculate the average time per response
    gjennomsnittligtid = tid / antallspm

    # Write the average response time to the results file
    with open("RAG_results.txt", "a", encoding="utf-8", errors='replace') as file:
        file.write(f"Average response time: {gjennomsnittligtid}")
        file.write("\n")

if __name__ == "__main__":
    run_all_tests()