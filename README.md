# **BudgetGPT - AI Assistant for Nepal's Fiscal Data**

**BudgetGPT** is an AI-powered assistant designed to help users easily access and understand Nepal's budget speech and fiscal allocations through natural language queries. Whether you're looking for budget details in Nepali or English, **BudgetGPT** provides quick, accurate answers.

---

## 💡 **Features**

- **Multi-language Support**: Understand and respond to queries in both Nepali and English.
- **Instant Budget Insights**: Get answers to questions related to budget allocations, sectors, and reforms from official budget documents.
- **AI-Powered**: Uses advanced language models and document retrieval systems to provide highly accurate and contextually relevant responses.
- **User-friendly Web Interface**: Interact with the assistant via an intuitive web interface.

---

## 🛠️ **Technologies Used**

- **Backend Framework**: **Flask** for building the web application.
- **Natural Language Processing (NLP)**: Utilizes **Langchain**, **HuggingFace embeddings**, and models like **CTransformers** and **LlamaCpp** for text generation.
- **Document Indexing & Retrieval**: **FAISS** for efficient and fast similarity-based search.
- **Language Detection**: **langdetect** to identify the language of the user query.

---

## 🌐 **How It Works**

1. **User Query**: The user inputs a query about Nepal's budget.
2. **Language Detection**: The system detects if the query is in Nepali or English.
3. **AI Model**: The corresponding language model (English or Nepali) is used to retrieve relevant data from indexed budget documents.
4. **Answer Generation**: The model provides an accurate response based on the data, ensuring no fabricated information is presented.

---

## 📂 **Technologies & Models Used**

- **Embedding Models**:
  - **English**: *BGE-M3* (HuggingFace)
  - **Nepali**: *All-MiniLM-L6-v2-Nepali* (HuggingFace)
  
- **Text Generation Models**:
  - **English**: *Mistral-7B-Instruct* (via CTransformers)
  - **Nepali**: *Nepali-LLM-Instruct* (available via HuggingFace)
  
- **Document Storage and Search**: **FAISS** for fast document retrieval and similarity-based search.

---

## 💬 **Example Queries**

- "What’s the budget for agriculture in FY 2024/25?"
- "गृह मन्त्रालयको बजेट कति हो?"
- "Key infrastructure projects in the road development sector in 2024/2025"
- "कृषि र पशुपन्छी विकास मन्त्रालयको लागि आर्थिक वर्ष २०८१/८२ को बजेट कति हो?"

---

## 📑 **Data Sources**

The data source for BudgetGPT is the **Ministry of Finance's Budget Speech**:

- **English**: Budget data is directly extracted and indexed from official **PDF documents**.
- **Nepali**: Budget data is parsed from a **text file** (`Nep_Budget_speech.txt`) that contains the speech content, as the direct parsing of Nepali text from PDFs is not feasible due to limitations.

---
## 📂 **Model Credits**

- **Nepali Embedding Model**: *All-MiniLM-L6-v2-Nepali* by **Sanjaya Subedi** ([Source](https://huggingface.co/jangedoo/all-MiniLM-L6-v2-nepali))
- **Nepali Text Generation Model**: *Nepali-LLM-Instruct* by **Shivam Sourav** ([Source](https://huggingface.co/shivam9980/NEPALI-LLM-INSTRUCT-Q4_K_M-GGUF))

## 📬 **Contact**

For feedback or inquiries, reach out to **aryalprashant3@gmail.com**.

