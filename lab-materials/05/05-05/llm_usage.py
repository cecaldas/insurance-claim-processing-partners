import os

# Imports
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator

INFERENCE_SERVER_URL = "http://llm.ic-shared-llm.svc.cluster.local:11434"
MAX_NEW_TOKENS = 96
TOP_K = 10
TOP_P = 0.95
TYPICAL_P = 0.95
TEMPERATURE = 0.9
REPETITION_PENALTY = 1.03
MODEL = "mistral"

def infer_with_template(input_text, template):
    # LLM definition
    llm = Ollama(
        base_url=inference_server_url,
        model="mistral",
        top_p=0.92,
        temperature=0.01,
        num_predict=512,
        repeat_penalty=1.03,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    PROMPT = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    return llm_chain.run(input_text)

def similarity_metric(predicted_text, reference_text):
    embedding_model = HuggingFaceEmbeddings()
    evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)
    distance_score = evaluator.evaluate_strings(prediction=predicted_text, reference=reference_text)
    return 1-distance_score["score"]
