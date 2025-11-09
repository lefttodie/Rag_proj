from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Sequence, Annotated, Dict, List
from dotenv import load_dotenv, find_dotenv
import os
import json

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader, PyPDFLoader, WebBaseLoader
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore



os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64)"


load_dotenv(find_dotenv())

# LLm Model Define (That's a Google free LLm Model in case Free Quota is reach the limit we can use the Groq model, just comment the model.)
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     temperature=1
# )

# LLM Model Define (That's a free Groq model in case Quota is reach the limit we can use the Google model, just comment the model.)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,
    max_tokens=100,
    # api_key=groq_api_key
)

# Embedding Model Define
embeddings = HuggingFaceEmbeddings(
    model = "BAAI/bge-small-en-v1.5",
)



with open("rbi.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    Document(
        page_content=item["content"],
        metadata={
            "title": item.get("title", "Untitled"),
            "url": item.get("url", "Unknown")
        }
    )
    for item in data if "content" in item
]




# Step 3: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
def clean_text(text: str) -> str:
    # Remove extra empty lines and excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


docs = text_splitter.split_documents(documents)
cleaned_docs = [
    doc.__class__(page_content=clean_text(doc.page_content), metadata=doc.metadata)
    for doc in docs
]

# Supabase Vectorstore
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Check if table exists and has data
try:
    # First, let's check the current state
    response = supabase.table("documents").select("count", count="exact").execute()
    print(f"Current documents in table: {response.count}")
    
    # If table is empty, populate it
    if response.count == 0:
        print("Creating Supabase vector store with documents...")
        vectorstore = SupabaseVectorStore.from_documents(
            documents=cleaned_docs,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        print("Supabase vector store created successfully")
        
        # Verify documents were added
        response = supabase.table("documents").select("count", count="exact").execute()
        print(f"Documents after creation: {response.count}")
    else:
        print("Table already has documents, connecting to existing vector store...")
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )

except Exception as e:
    print(f"Error with Supabase vector store: {str(e)}")
    # Fallback to FAISS for testing
    print("Falling back to FAISS for testing...")
    vectorstore = FAISS.from_documents(cleaned_docs, embeddings)

# Create retriever from vectorstore
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)





# Defineing the tool
@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from resource as per requirements."""

    # docs1 = retriever.invoke(query)
    docs1 = retriever.get_relevant_documents(query)
    #So here, we’re doing a similarity search
    if not docs1:
        return "I found no relevant information."
    #If FAISS returns nothing, the function handles it gracefully
    results = []
    for i, doc in enumerate(docs1):
        content = clean_text(doc.page_content)
        preview = content[:1000] + "..." if len(content) > 1000 else content
        results.append(f"Document {i+1}:\n{preview}")
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

# Define the state structure of your agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state: AgentState):
    """Check if the message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls) > 0

# Prompt Defining
system_prompt = """
You are a professional, clear, and empathetic AI assistant whose expertise for this session is limited to a curated dataset of RBI-related documents, stored in a vector database. All information must be fetched via the RAG pipeline from the embedded PDF chunks.

Tone & style:
- Be concise, human, and professional. Use empathetic language when appropriate (e.g., "I understand your concern").
- Do not use emojis or casual slang.
- Do NOT start answers with mechanical phrases like "According to the dataset." Integrate source information naturally when needed.

Rules of operation:
1. Questions about RBI policies, reports, guidelines, or financial regulations:
   - ALWAYS call the `retriever_tool` to fetch relevant passages before composing your answer.
   - Answer strictly using only the retrieved content. Do not invent or infer additional facts.
   - Present answers clearly: 1–2 sentence summary first, then bullet points or numbered details if helpful.
   - Include a short natural citation if available (e.g., report title or section).
   - If the retriever returns no useful content, respond politely with:
     "I’m sorry — I don’t have information about that in my RBI dataset. Please ask another question or rephrase."

2. General conversation / greetings (e.g., "Hello", "How are you?", "Thanks"):
   - Respond naturally and politely without using the RBI dataset. Keep replies brief and friendly.

3. Questions outside RBI scope:
   - Reply professionally and briefly with:
     "I’m sorry — I don’t have information about that topic in my RBI dataset. My scope is limited to RBI reports, policies, and related financial regulations."
   - Do not attempt to answer using outside knowledge.

Try to featch or find out the answer from the vectordb from this types of questions:
1. What is a Non-Banking Financial Company (NBFC)?
2. What does conducting financial activity as “principal business” mean?
3. NBFCs are doing functions similar to banks. What is the difference between banks and NBFCs?
4. Is it necessary that every NBFC should be registered with the Reserve Bank?
5. What are the requirements for registration with the Reserve Bank?
6. What is the procedure for application to the Reserve Bank for Registration?
7. What are the essential documents required to be submitted along with the application form to the Reserve Bank?
8. What is Scale Based Regulatory Framework or SBR Framework for NBFCs?
9. Does the Reserve Bank regulate all financial companies?
10. What are the different types/categories of NBFCs registered with the Reserve Bank?
11. What are the powers of the Reserve Bank with regard to 'Non-Bank Financial Companies’, that is, companies that meet the Principal Business Criteria or 50-50 criteria?
12. What action can be taken against persons/financial companies making false claim of being regulated by the Reserve Bank?
13. What action is taken if financial companies which are lending or making investments as their principal business do not obtain a Certificate of Registration from the Reserve Bank?

   
Safety and clarity:
- Always ensure answers are grounded in retrieved content from your vector database.
- Avoid over-citation; use concise citations if necessary.
- Be explicit about limitations when uncertain.

Response format example for in-scope question:
- Short summary sentence.
- Bullet list of key points (policy details, regulations, guidelines, etc.).
- Citation line: "Source: <Report Title / Section>"


Most import thing, while responding make shure no respond should be perfect in text format, it should not give the output with function name and all 

Goal:
Be human, helpful, and strictly grounded in the RBI PDF dataset via your RAG retriever, while allowing normal polite conversation.
"""



# Creating a dictionay of out tools
tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLm Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])   # get chat history
    messages = [SystemMessage(content=system_prompt)] + messages  # prepend system prompt
    messages = llm.invoke(messages)   # call the LLM with system+chat history
    return {"messages": [messages]} # return new state or make a copy of it into messages.



# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response"""
    # look at tool calls made by the LLM → execute them → return the results as messages that the LLM can consume in the next step.
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provide')}")

        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(result))}")

        # Append the tool meassage
        results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
    
    print("Tool Execution Complete. Back to the model!")
    return {"messages": results}


# Defining the Graph
graph = StateGraph(AgentState)

# Adding Nodes
graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)

# Creating Conditional Edge
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever_agent",False:END
    }
)

# Creating edge
graph.add_edge("retriever_agent","llm")

# Defining entry point
graph.set_entry_point("llm")

# Graph compile
rag_agent = graph.compile()

# Fastapi
app = FastAPI(title="Rbi RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        messages = [HumanMessage(content=request.question)]
        result = rag_agent.invoke({"messages": messages})
        answer = result['messages'][-1].content
        return QueryResponse(answer=answer)
    except Exception as e:
        return QueryResponse(answer=f"Error: {str(e)}")
    


