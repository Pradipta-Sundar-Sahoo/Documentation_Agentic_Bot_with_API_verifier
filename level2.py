from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
import json
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from openai import OpenAI
import os
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
import uuid
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
import uuid
from datetime import datetime
from vector_db import update_vector_db


from dotenv import load_dotenv
load_dotenv()



OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
SERPER_API_KEY = os.environ['SERPER_API_KEY']
LLAMA_CLOUD_API_KEY=os.environ['LLAMA_CLOUD_API_KEY']

client = OpenAI(api_key=OPENAI_API_KEY)



@tool
def query_vectordb(query: str):
    """
    Query information about Crustdata API usage and endpoints.
    Args:
        query (str): The query string about Crustdata API.
    Returns:
        str: Detailed response about API usage.
    """
    # Load database
    db = FAISS.load_local("crustdata_db", OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    
    # Get similar documents
    docs = db.similarity_search(query, k=4)
    
    # Create context from documents
    context = " ".join(doc.page_content for doc in docs)
    
    # Structured prompt template
    prompt = f"""You are a Crustdata API expert helping developers understand and implement Crustdata's API.

    Context: {context}

    User Query: {query}

    Most Important: 
    1. Give the Curl / Python commands wherever you think you can give wrt query.
    2. Give elaborative response, as if you are making understand a baby.

    Please address the following:
    1. The endpoint's purpose, features, and authentication details.
    2. Key implementation details like parameters and response structure.
    3. Best practices, usage limits, and performance tips.
    4. Common errors, troubleshooting, and limitations.
    5. Always try to give the source where you are refering(see if there's any link, python documentation, curl command)

    Use only the information from the context and mention if anything is missing."""


    # Use OpenAI client directly
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"Given the following context about Crustdata API: {context},(Give the Curl / Python commands wherever you think you can give wrt query.) answer the following question: {query}. {prompt}"
        }],
        temperature=0.9,
        max_tokens=4096,
    )
    
    return response.choices[0].message.content



@tool
def query_google_search(query: str):
    """
    Perform a Google search using the GoogleSerper API and retrieve search results based on the query.

    Args:
        query (str): The search query to be processed.

    Returns:
        str: The response from the GoogleSerper API containing search results.
    """
    search = GoogleSerperAPIWrapper()
    new_query = "Answer clearly , " + query
    return search.run(new_query)

import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
import re
from langchain_core.tools import BaseTool

class APIEndpointValidator(BaseTool):
    name: str = "validate_api_endpoint"
    description: str = "Validates if a given URL is a valid API endpoint by attempting a HEAD request."
    
    def _run(self, url: str) -> str:
        """
        The internal run method called by invoke().
        """
        try:
            response = requests.head(url, timeout=5)
            if response.status_code < 400:
                return f"Valid API endpoint (Status: {response.status_code})"
            else:
                return f"Invalid API endpoint (Status: {response.status_code})"
        except requests.exceptions.RequestException as e:
            return f"Failed to validate API endpoint: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        return self._run(url)

def extract_api_urls_with_llm(text: str, llm: ChatOpenAI) -> List[str]:
    """
    Uses LLM to extract API endpoints from text, including those in curl commands.
    
    Args:
        text (str): Text to analyze
        llm (ChatOpenAI): LLM instance to use for extraction
    Returns:
        List[str]: List of found API URLs
    """

    prompt = """Extract all API endpoints from the following text. Look for:
    1. Direct URLs starting with https://api.crustdata.com
    2. URLs within curl commands
    3. URLs within code snippets or examples
    
    Return only the full URLs, one per line. If no valid API endpoints are found, return "None".
    
    Text to analyze:
    {text}
    
    API Endpoints:"""
    
    response = llm.invoke(prompt.format(text=text))
    
    # Process the response to extract URLs
    urls = []
    for line in response.content.split('\n'):
        # Clean up the line
        line = line.strip()
        if line and line.lower() != "none":
            # Extract URL if it matches our pattern
            matches = re.findall(r'https://api\.crustdata\.com[^\s<>"]*', line)
            urls.extend(matches)
    
    return list(set(urls))

class APIValidatorNode:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.validator = APIEndpointValidator()
        self.llm = llm or ChatOpenAI(
            temperature=0,
            model="gpt-4"
        )
        
    def __call__(self, state: Dict) -> Dict:
        """
        Process the state and validate any Crustdata API endpoints found in tool responses.
        """
        messages = state.get("messages", [])
        if not messages:
            return state
        
        # Get the latest message content
        latest_message = messages[-1]
        content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # Extract API URLs using LLM
        api_urls = extract_api_urls_with_llm(content, self.llm)
        validation_results = []
        
        for url in api_urls:
            result = self.validator.invoke(url)
            validation_results.append(f"Crustdata API URL: {url}\nValidation: {result}")
        
        if validation_results:
            validation_summary = "\n\nCrustdata API Validation Results:\n" + "\n".join(validation_results)
            
            if hasattr(latest_message, 'content'):
                latest_message.content += validation_summary
            else:
                messages[-1] = str(latest_message) + validation_summary
        
        return {"messages": messages}

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]



class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

primary_assistant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Emma, a Crustdata AI assistant supervisor. 
            You are a tool selection expert. Analyze the query and select the most appropriate tool:
            1. query_vectordb - For Crustdata API specific information
            2. query_google_search - For general information
            
            Select based on:
            - Query specificity to Crustdata
            - Required information source
            - Query complexity

            JUST go the tool , Get me the most appropriate response and write it.

            For each question:
            The Provided context will be a Crustdata company's API usage queries with multiple steps/procedures.

            Directly retrieve relevant information from the context provided.
            Answer only with the requested details.
            Keep responses clear and to the point without extra information.
            Give responses in detail and to the point.

            don't write any special characters or symbols like *. Just simple plain English
            dont need to bold/italic any characters/words"""
        " You are an intelligent supervisor who knows which tool will be required for the Query provided "
        " Use the provided tools to query_vectordb, query_google_search, and other information to assist the user's queries. "
    ),
    ("placeholder", "{messages}"),
])

def create_graph():

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4"
    )
    
    part_1_tools = [
        query_google_search,
        query_vectordb,
        APIEndpointValidator()
    ]
    part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

    builder = StateGraph(State)

    builder.add_node("assistant", Assistant(part_1_assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
    builder.add_node("api_validator", APIValidatorNode(llm=llm))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "api_validator")
    builder.add_edge("api_validator", "assistant")
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)



@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime

class ChatSession:
    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.session_id = str(uuid.uuid4())
    
    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        ))
    
    def get_context_window(self, window_size: int = 5) -> List[ChatMessage]:
        """Returns the last n messages for context"""
        return self.messages[-window_size:]

def init_session_state():
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_session' not in st.session_state:
        st.session_state.current_session = ChatSession()
        st.session_state.chat_sessions[st.session_state.current_session.session_id] = st.session_state.current_session
def get_final_response_with_memory(graph, question: str, session: ChatSession):
    context = session.get_context_window()
    context_str = "\n".join([
        f"{msg.role}: {msg.content}" 
        for msg in context
    ])
    
    enriched_question = f"""Previous conversation:\n{context_str}\n\nCurrent question: {question}"""
    
    thread_id = str(uuid.uuid4())
    config = {"thread_id": thread_id}
    
    response = graph.invoke(
        {
            "messages": ("user", enriched_question)
        },
        config
    )
    
    if response.get("messages"):
        final_message = response["messages"]
        if isinstance(final_message, list):
            final_message = final_message[-1]
        
        content = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        session.add_message("user", question)
        session.add_message("assistant", content)
        
        return [{
            "question": question,
            "response": content
        }]
    return []

# Update the display function to show tool selection more clearly
def display_chat_history(session: ChatSession):
    for msg in session.messages:
        if msg.role == "user":
            htmlstr1 = f"""
                <p style='background-color:rgb(0, 100, 200, 1);
                        color: rgb(255, 255, 255, 1);
                        font-size: 14px;
                        border-radius: 7px;
                        padding-left: 12px;
                        padding-top: 13px;
                        padding-bottom: 13px;
                        line-height: 25px;'>
                    👤: {msg.content}
                </p>
                """
            st.markdown(htmlstr1, unsafe_allow_html=True)
        else:
            # Parse the content to extract tool information if available
            st.success(f"🤖 : {msg.content}")

import streamlit as st
import os
from bs4 import BeautifulSoup
import requests
from llama_parse import LlamaParse
from datetime import datetime
import validators

def setup_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists("data"):
        os.makedirs("data")

def save_to_txt(content: str, filename: str) -> str:
    """Save content to a txt file in the data directory"""
    setup_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{filename}_{timestamp}.txt"
    filepath = os.path.join("data", safe_filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Update vector database after saving new file
    update_vector_db()
    
    return filepath

def process_url(url: str) -> str:
    """Process URL using BeautifulSoup"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        return text
    
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return ""

def process_pdf(uploaded_file) -> str:
    """Process PDF using LlamaParse"""
    try:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="en"
        )
        extra_info = {"file_name": uploaded_file.name}
        file_bytes = uploaded_file.read()
        documents = parser.load_data(file_bytes, extra_info=extra_info)
        return "\n".join([doc.text for doc in documents])
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def sidebar_file_upload():
    """Add file upload functionality to the sidebar"""
    st.sidebar.markdown("### Upload Document")
    
    # URL input
    url = st.sidebar.text_input("Enter URL:")
    if url and st.sidebar.button("Process URL"):
        if validators.url(url):
            with st.sidebar.spinner("Processing URL..."):
                content = process_url(url)
                if content:
                    filename = url.split('/')[-1] or "webpage"
                    filepath = save_to_txt(content, filename)
                    st.sidebar.success(f"URL processed and saved to {filepath}")
        else:
            st.sidebar.error("Please enter a valid URL")


def main():
    st.title("Crustdata Agentic Chat Interface")
    
    # Initialize session state
    init_session_state()
    
    # Session management
    if st.sidebar.button("New Chat Session"):
        st.session_state.current_session = ChatSession()
        st.session_state.chat_sessions[st.session_state.current_session.session_id] = st.session_state.current_session
        st.rerun()
    st.sidebar.markdown("### To Update Database, \n enter valid URL or pdf")
    url = st.sidebar.text_input("Enter URL:")
    if url and st.sidebar.button("Process URL"):
        if validators.url(url):
            with st.spinner("Processing URL..."):
                content = process_url(url)
                if content:
                    filename = url.split('/')[-1] or "webpage"
                    filepath = save_to_txt(content, filename)
                    st.sidebar.success(f"URL processed and database updated at {filepath}")
        else:
            st.sidebar.error("Please enter a valid URL")
    
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])
    if pdf_file is not None and st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            content = process_pdf(pdf_file)
            if content:
                filename = pdf_file.name.rsplit('.', 1)[0]
                filepath = save_to_txt(content, filename)
                st.sidebar.success(f"PDF processed and database updated at {filepath}")
    
    # Display processed files
    if os.path.exists("data"):
        st.sidebar.markdown("### Processed Files")
        files = os.listdir("data")
        if files:
            for file in files:
                st.sidebar.text(f"• {file}")
        else:
            st.sidebar.text("No processed files yet")

    # Display available sessions
    session_ids = list(st.session_state.chat_sessions.keys())
    if len(session_ids) > 1:
        selected_session = st.sidebar.selectbox(
            "Select Chat Session",
            session_ids,
            index=session_ids.index(st.session_state.current_session.session_id),
            format_func=lambda x: f"Session {session_ids.index(x) + 1}"
        )
        st.session_state.current_session = st.session_state.chat_sessions[selected_session]
    
    st.write("Current Session ID:", st.session_state.current_session.session_id)

    display_chat_history(st.session_state.current_session)

    question = st.chat_input("Enter your question:")
    
    if question:

        graph = create_graph()
        results = get_final_response_with_memory(graph, question, st.session_state.current_session)
        
        # Display the results
        for result in results:
            st.markdown("-" * 80)
            st.markdown(f"**Question**: \n {result['question']}")
            st.success(f"Response: \n {result['response']}")
            st.markdown("-" * 80)

if __name__ == "__main__":
    main()