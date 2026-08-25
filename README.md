Multi-Agent Technical RAG Assistant (LangChain & Qdrant)
An intelligent, double-verified Retrieval-Augmented Generation (RAG) assistant built with LangGraph, Qdrant Cloud, and Streamlit. The system ingests official LangChain and Qdrant technical documentation and uses a two-agent workflow to answer technical queries accurately while strictly preventing hallucinations.

🏗 System Architecture & Agent Workflow
The system uses a stateful multi-agent loop to enforce strict groundedness:

[User Query] ➡️ [Agent 1: Researcher] ➡️ [Agent 2: Reviewer] ➡️ [Streamlit Interface]
                      │                          │
              Reads from Qdrant           Audits Draft Answer
             (Retrieves Chunks)          (APPROVE or REJECT)
Agent 1 – Researcher: Queries the Qdrant vector database for relevant semantic chunks, fetches exact source URLs, and generates a draft answer grounded only in the retrieved context.

Agent 2 – Reviewer: Evaluates the draft answer against the raw context chunks. Approves supported answers (APPROVED) or triggers automated revision/refusal if any claim lacks source backing.

Out-of-Scope Guardrail: If no relevant documentation chunks are found for general or off-topic queries, the system strictly refuses to answer.

🧪 Sample Test Cases & Expected Agent Behavior
Below are 5 representative test queries demonstrating how the two-agent system processes in-scope technical questions versus out-of-scope refusal guardrails.

Case 1: Technical Query — Document Loading (In-Scope)
Query: "How does LangChain handle document loading?"

Retrieved Source: [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)

Agent Output:

Based on the provided documentation, document loading is handled by fetching content (such as markdown pages using requests.get()) and converting the retrieved text and its metadata into a list of LangChain Document objects.

Reviewer Verdict: APPROVED (Answer is fully supported by context).

Case 2: Technical Query — Text Chunking (In-Scope)
Query: "What is the function of RecursiveCharacterTextSplitter in LangChain?"

Retrieved Source: [https://python.langchain.com/docs/how_to/recursive_text_splitter/](https://python.langchain.com/docs/how_to/recursive_text_splitter/)

Agent Output:

RecursiveCharacterTextSplitter splits long text documents into smaller chunks hierarchically using a default set of separators (["\n\n", "\n", " ", ""]). It attempts to keep paragraphs, sentences, and words together to preserve semantic context within specified chunk_size and chunk_overlap boundaries.

Reviewer Verdict: APPROVED (Answer is fully supported by context).

Case 3: Technical Query — Qdrant Setup (In-Scope)
Query: "How do I initialize a Qdrant vector store in Python?"

Retrieved Source: [https://python.langchain.com/docs/integrations/vectorstores/qdrant/](https://python.langchain.com/docs/integrations/vectorstores/qdrant/)

Agent Output:

You initialize a Qdrant vector store by instantiating QdrantClient with your database URL and API key, and passing it along with an embedding model into QdrantVectorStore(client=client, collection_name="your_collection", embedding=embeddings).

Reviewer Verdict: APPROVED (Answer is fully supported by context).

Case 4: General Knowledge Refusal (Out-of-Scope)
Query: "Who won the 2022 FIFA World Cup?"

Retrieved Source: None

Agent Output:

I cannot answer this query based on the provided documentation.

Reviewer Verdict: APPROVED (Refusal verified; query is outside vector database scope).

Case 5: Out-of-Scope Trivia Refusal (Out-of-Scope)
Query: "What is the capital of France?"

Retrieved Source: None

Agent Output:

I cannot answer this query based on the provided documentation.

Reviewer Verdict: APPROVED (Refusal verified; query is outside vector database scope).

🛠 Tech Stack
Component	Technology
Agent Orchestration	LangGraph / LangChain
Vector Database	Qdrant Cloud
Embeddings	FastEmbed (BAAI/bge-small-en-v1.5)
Text Processing	RecursiveCharacterTextSplitter (chunk size: 1000, overlap: 150)
User Interface	Streamlit
Language Model	Google Gemini API / Groq
⚙️ Setup & Installation
1. Clone Repository & Setup Virtual Environment
Bash
git clone <YOUR_GITHUB_REPO_URL>
cd sprint-R3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Configure Environment Variables
Create a .env file in the root directory using .env.example as a template:

Code snippet
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
📥 Vector Data Ingestion
To scrape, chunk, embed, and upload the LangChain & Qdrant documentation into your remote Qdrant collection:

Bash
python ingest.py
🖥 Running the Streamlit App
Launch the interactive web interface:

Bash
streamlit run app.py
Access the application in your browser at http://localhost:8501.

📁 Repository Structure
├── app.py              # Streamlit Web UI displaying answers, sources & reviewer verdict
├── graph.py            # LangGraph multi-agent orchestration state graph
├── ingest.py           # Scraping, chunking, and Qdrant vector store ingestion pipeline
├── .env.example        # Environment variable template (No actual keys committed)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation & test case specifications