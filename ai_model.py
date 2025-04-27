from document_parser import parse_budget_speech_pdf_en, parse_budget_speech_txt_np
from config import MODEL_PATH_EN, MODEL_PATH_NP
from config import TXT_PATH_NP, PDF_PATH_EN
from config import INDEX_PATH_EN,INDEX_PATH_NP
import os
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma

def initialize_ai():
    embeddings_en = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    embeddings_np = HuggingFaceEmbeddings(
        model_name="jangedoo/all-MiniLM-L6-v2-nepali",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(INDEX_PATH_EN):
        vector_store_en = FAISS.load_local(INDEX_PATH_EN, embeddings_en, allow_dangerous_deserialization=True)
    else:
        documents = parse_budget_speech_pdf_en(PDF_PATH_EN)
        vector_store_en = FAISS.from_documents(documents, embeddings_en)
        vector_store_en.save_local(INDEX_PATH_EN)

    if os.path.exists(INDEX_PATH_NP):
        vector_store_np = FAISS.load_local(INDEX_PATH_NP, embeddings_np, allow_dangerous_deserialization=True)
    else:
        documents = parse_budget_speech_txt_np(TXT_PATH_NP)
        vector_store_np = FAISS.from_documents(documents, embeddings_np)
        vector_store_np.save_local(INDEX_PATH_NP)

    # *********************** chromadb********************
    #  if os.path.exists(INDEX_PATH_EN) and os.path.isdir(INDEX_PATH_EN):
    #     vector_store = Chroma(
    #         persist_directory=INDEX_PATH_EN,
    #         embedding_function=embeddings_en
    #     )
    # else:
    #     documents = parse_budget_speech_pdf_en(PDF_PATH_EN)
    #     vector_store = Chroma.from_documents(
    #         documents=documents,
    #         embedding=embeddings_en,
    #         persist_directory=INDEX_PATH_EN
    #     )

    llm_en = CTransformers(
        model=MODEL_PATH_EN,
        model_type="mistral",
        config={'max_new_tokens': 512,
                'temperature': 0.2,
                'context_length': 2048,
                'gpu_layers': 0}
    )
    llm_np = LlamaCpp(
        model_path=MODEL_PATH_NP,
        temperature=0.4,
        max_tokens=512,
        verbose=False
    )
    return vector_store_en,vector_store_np, llm_en, llm_np

# Initializing AI components
vector_store_en,vector_store_np,llm_en,llm_np = initialize_ai()
retriever_en = vector_store_en.as_retriever(search_kwargs={"k": 2})
retriever_np = vector_store_np.as_retriever(search_kwargs={"k": 1})

# English prompt template
prompt_template_en = """
    Refer to Nepal's budget data:
    {context}

    Question: {question}
    Provide a concise, clear, fluent and easily readable response based strictly on the provided budget information. If allocation amounts are specified, include them accurately. Ensure the answer is factual and grounded in the data—do not infer or add details beyond what's given.

    Answer:
    """

QA_PROMPT_EN = PromptTemplate(
    template=prompt_template_en, input_variables=["context", "question"]
)

# Nepali prompt template
prompt_template_np = """
        निम्न सन्दर्भको आधारमा सोधिएको बजेट सम्बन्धी प्रश्नको विस्तृत, स्पष्ट र व्यवस्थित उत्तर दिनुहोस्।
        उत्तर नेपाली भाषामै मात्र दिनुहोस्। 
        सन्दर्भ बाहिरको जानकारी प्रयोग नगर्नुहोस्।

        सन्दर्भ:
        {context}

        प्रश्न:
        {question}

        उत्तर:
        """

QA_PROMPT_NP  = PromptTemplate(
    template = prompt_template_np, input_variables=["context", "question"]
)

# The complete Q&A pipeline:
# 1. Searches CV content (retriever)
# 2. Formats question + context (QA_PROMPT)
# 3. Generates answer (llm)

qa_chain_en = RetrievalQA.from_chain_type(
    llm=llm_en,
    chain_type="stuff",
    retriever=retriever_en,
    chain_type_kwargs={"prompt": QA_PROMPT_EN},
    return_source_documents=True
)

qa_chain_np = RetrievalQA.from_chain_type(
    llm=llm_np,
    chain_type="stuff",
    retriever=retriever_np,
    chain_type_kwargs={"prompt": QA_PROMPT_NP},
    return_source_documents=True
)
