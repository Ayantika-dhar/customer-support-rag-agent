import streamlit as st
import os
import sys

# Make sure Python can find models/ and config/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_chatgroq_model
from models.embeddings import EmbeddingClient
from utils.rag import build_knowledge_base
from utils.assistant import answer_query


def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")

    st.markdown(
        """
    ## 🔧 Installation

    First, install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    ## API Key Setup

    You'll need API keys from your chosen provider. Get them from:

    ### OpenAI
    - Visit OpenAI Platform
    - Create a new API key
    - Set the variables in config

    ### Groq
    - Visit Groq Console
    - Create a new API key
    - Set the variables in config

    ### Google Gemini
    - Visit Google AI Studio
    - Create a new API key
    - Set the variables in config

    ## How to Use

    1. Go to the Chat page (use the navigation in the sidebar)
    2. Build the Knowledge Base
    3. Start chatting once everything is configured!
    """
    )


def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")

    # Load selected response mode from sidebar
    mode = st.session_state.get("mode", "Concise")

    # Get Groq chat model
    chat_model = get_chatgroq_model()

    # If no model (e.g., no API key), show info and exit
    if chat_model is None:
        st.info(
            "🔧 No Groq API key found. Please set GROQ_API_KEY in your environment, then reload the app."
        )
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (RAG + Web Search pipeline)
    if prompt := st.chat_input("Ask something..."):

        # Store user message
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if Knowledge Base is built
        if "embed_client" not in st.session_state or "vectorstore" not in st.session_state:
            with st.chat_message("assistant"):
                st.error("❌ Knowledge Base not built yet. Click '📚 Build Knowledge Base' in the sidebar.")
            return

        embed_client = st.session_state["embed_client"]
        vectorstore = st.session_state["vectorstore"]

        # Get RAG + Web Search answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_query(
                    user_query=prompt,
                    mode=mode.lower(),
                    chat_model=chat_model,
                    embed_client=embed_client,
                    vectorstore=vectorstore,
                )

                st.markdown(result["answer"])

                # Show sources in an expander
                with st.expander("📄 Sources Used"):
                    if result["rag_results"]:
                        st.markdown("### 🔧 Internal Docs")
                        for r in result["rag_results"]:
                            st.write(f"- **{r['source']}** (score: {r['score']:.2f})")

                    if result["web_results"]:
                        st.markdown("### 🌐 Web Results")
                        for w in result["web_results"]:
                            st.write(f"- [{w['title']}]({w['url']})")

        # Add assistant response to chat history
        st.session_state["messages"].append(
            {"role": "assistant", "content": result["answer"]}
        )


def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

        st.divider()

        # Response Mode Toggle
        mode = st.radio(
            "Response Mode:",
            ["Concise", "Detailed"],
            index=0,
        )
        # Store mode in session_state so chat_page can read it
        st.session_state["mode"] = mode

        st.divider()

        # Build Knowledge Base Button
        st.markdown("### Knowledge Base")
        if st.button("📚 Build Knowledge Base"):
            try:
                embed_client = EmbeddingClient()
                vectorstore = build_knowledge_base("data/docs", embed_client)

                st.session_state["embed_client"] = embed_client
                st.session_state["vectorstore"] = vectorstore
                st.success("Knowledge Base built successfully!")
            except Exception as e:
                st.error(f"Error building KB: {e}")

        st.divider()
        if page == "Chat":
            if st.button("🗑 Clear Chat History", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
