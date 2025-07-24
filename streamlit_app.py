#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py - IoT Security Intelligent Assistant Streamlit Web Interface
Web chat interface based on existing RAG Agent implementation
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, List
import traceback

# Import existing Agent (no modifications)
try:
    from rag_agent import SimpleRAGAgent
    from ollama_client import OllamaClient
except ImportError as e:
    st.error(f"❌ Import failed: {e}")
    st.error("Please ensure all necessary files exist in the same directory")
    st.stop()


class StreamlitWebInterface:
    """Streamlit Web Interface Manager"""

    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="IoT Security Intelligent Assistant",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        """Initialize session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "agent" not in st.session_state:
            st.session_state.agent = None

        if "agent_config" not in st.session_state:
            st.session_state.agent_config = {
                "model_name": "llama3",
                "base_url": "http://localhost:11434",
                "timeout": 120,
                "max_tokens": 800,
                "temperature": 0.7,
                "use_rag": True,
                "rag_top_k": 3
            }

        if "system_status" not in st.session_state:
            st.session_state.system_status = None

        if "show_settings" not in st.session_state:
            st.session_state.show_settings = False

    def create_sidebar(self):
        """Create sidebar"""
        with st.sidebar:
            st.title("🛡️ IoT Security Assistant")
            st.markdown("---")

            # System status
            # self.show_system_status()
            # st.markdown("---")

            # Settings panel
            self.show_settings_panel()
            st.markdown("---")

            # Action buttons
            self.show_action_buttons()
            st.markdown("---")

            # Usage guide
            self.show_usage_guide()

    def show_system_status(self):
        """Display system status"""
        st.subheader("📊 System Status")

        # Check Agent status
        if st.session_state.agent is None:
            st.error("🔴 Agent not initialized")
            if st.button("🔧 Initialize Agent", key="init_agent"):
                self.initialize_agent()
                st.rerun()
        else:
            # Get system status
            try:
                status = st.session_state.agent.get_system_status()
                st.session_state.system_status = status

                # Ollama status
                ollama_status = status["ollama"]["available"]
                st.success("🟢 Ollama Service") if ollama_status else st.error("🔴 Ollama Service")

                # Model information
                st.info(f"🤖 Model: {status['ollama']['model']}")

                # RAG status
                rag_enabled = status["rag"]["enabled"]
                rag_status = "🟢 RAG Knowledge Base" if rag_enabled else "🟡 Basic Mode"
                st.success(rag_status) if rag_enabled else st.warning(rag_status)

                if rag_enabled:
                    st.caption(f"📚 Documents: {status['rag']['documents_loaded']}")

                # Performance statistics
                perf = status["performance"]
                st.metric("Total Queries", perf["total_queries"])
                st.metric("Average Confidence", f"{perf['average_confidence']:.1%}")

            except Exception as e:
                st.error(f"❌ Status retrieval failed: {e}")

    def show_settings_panel(self):
        """Display settings panel"""
        st.subheader("⚙️ Settings")

        # Expand/collapse settings
        if st.button("🔧 Adjust Parameters", key="toggle_settings"):
            st.session_state.show_settings = not st.session_state.show_settings

        if st.session_state.show_settings:
            with st.expander("Parameter Configuration", expanded=True):
                # Model settings
                temp_client = OllamaClient()
                models = temp_client.list_models()
                st.session_state.agent_config["model_name"] = st.selectbox(
                    "Model Selection",
                    models,
                    index=0,
                    key="model_select"
                )

                # Generation parameters
                st.session_state.agent_config["temperature"] = st.slider(
                    "Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1,
                    help="Higher values make responses more creative, lower values more conservative"
                )

                st.session_state.agent_config["max_tokens"] = st.slider(
                    "Maximum Response Length", 200, 2000, 800, 100,
                    help="Limit the maximum length of AI responses"
                )

                # RAG settings
                st.session_state.agent_config["use_rag"] = st.checkbox(
                    "Enable RAG Knowledge Base",
                    value=True,
                    help="Use local knowledge base to enhance response quality"
                )

                if st.session_state.agent_config["use_rag"]:
                    st.session_state.agent_config["rag_top_k"] = st.slider(
                        "Retrieved Documents", 1, 10, 3, 1,
                        help="Number of relevant documents retrieved from knowledge base"
                    )

                # Apply settings
                if st.button("✅ Apply Settings", key="apply_settings"):
                    self.initialize_agent()
                    st.success("✅ Settings updated")
                    time.sleep(1)
                    st.rerun()

    def show_action_buttons(self):
        """Display action buttons"""
        st.subheader("🎮 Actions")

        # Clear conversation
        if st.button("🗑️ Clear Conversation", key="clear_chat"):
            st.session_state.messages = []
            st.success("Conversation cleared")
            time.sleep(1)
            st.rerun()

        # Export conversation
        if st.button("💾 Export Conversation", key="export_chat"):
            self.export_conversation()

        # Reinitialize
        if st.button("🔄 Restart Agent", key="restart_agent"):
            st.session_state.agent = None
            self.initialize_agent()
            st.success("Agent restart completed")
            time.sleep(1)
            st.rerun()

    def show_usage_guide(self):
        """Display usage guide"""
        with st.expander("📖 Usage Guide"):
            st.markdown("""
            ### 🎯 Main Features

            **Threat Detection**
            - "IoT device abnormal network behavior analysis"
            - "How to detect DDoS attacks?"

            **Security Assessment**
            - "Assess smart home security risks"
            - "Industrial IoT network security assessment"

            **Protection Recommendations**
            - "How to protect IoT devices?"
            - "What are the protection measures?"

            **Research Summary**
            - "IoT encryption technology development trends"
            - "Latest research on IoT security"

            ### 💡 Usage Tips
            - The more specific the question description, the more accurate the answer
            - Can request step-by-step analysis
            - Supports both Chinese and English queries
            - Pay attention to confidence indicators
            """)

    def initialize_agent(self):
        """Initialize Agent"""
        try:
            with st.spinner("🔧 Initializing IoT Security Assistant..."):
                # Create Agent instance (using existing code, no modifications)
                agent = SimpleRAGAgent(st.session_state.agent_config)

                # Check Agent status
                status = agent.get_system_status()

                if not status["ollama"]["available"]:
                    st.error("❌ Ollama service unavailable")
                    st.error("Please ensure Ollama service is running: `ollama serve`")
                    return False

                st.session_state.agent = agent
                st.session_state.system_status = status

                # Add welcome message
                if not st.session_state.messages:
                    welcome_msg = {
                        "role": "assistant",
                        "content": "🛡️ Hello! I am an IoT security intelligent assistant that can help you analyze vulnerabilities, detect threats, and assess risks. May I help you with something?",
                        "timestamp": datetime.now(),
                        "metadata": {
                            "task_type": "welcome",
                            "confidence": 1.0,
                            "processing_time": 0.0
                        }
                    }
                    st.session_state.messages.append(welcome_msg)

                return True

        except Exception as e:
            st.error(f"❌ Agent initialization failed: {e}")
            st.error("Please check Ollama service and related dependencies")
            return False

    def show_chat_interface(self):
        """Display chat interface"""
        st.title("💬 IoT Security Intelligent Assistant")

        # Initialize Agent
        if st.session_state.agent is None:
            st.warning("⚠️ Initializing system...")
            if self.initialize_agent():
                st.rerun()
            else:
                st.stop()

        # Display chat history
        self.display_chat_history()

        # Quick question buttons
        self.show_quick_questions()

        # Input area
        self.show_input_area()

    def display_chat_history(self):
        """Display chat history"""
        # Create scrolling container
        chat_container = st.container()

        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    # User message
                    with st.chat_message("user", avatar="👤"):
                        st.write(message["content"])

                elif message["role"] == "assistant":
                    # AI response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message["content"])

                        # Display metadata (if available)
                        if "metadata" in message:
                            metadata = message["metadata"]

                            # Create column layout to display metrics
                            cols = st.columns(4)

                            with cols[0]:
                                if "confidence" in metadata:
                                    confidence = metadata["confidence"]
                                    st.metric("Confidence", f"{confidence:.1%}")

                            with cols[1]:
                                if "task_type" in metadata:
                                    task_type = metadata["task_type"]
                                    st.metric("Task Type", task_type)

                            with cols[2]:
                                if "processing_time" in metadata:
                                    proc_time = metadata["processing_time"]
                                    st.metric("Processing Time", f"{proc_time:.2f}s")

                            with cols[3]:
                                if "model_used" in metadata:
                                    model = metadata["model_used"]
                                    st.metric("Model", model)

                            # Display reference sources (if available)
                            if "sources" in metadata and metadata["sources"]:
                                with st.expander("📚 Reference Sources", expanded=False):
                                    for j, source in enumerate(metadata["sources"][:], 1):
                                        source_type = "📚 Academic Research" if source['type'] == 'academic' else "🚨 CVE Vulnerability"
                                        st.write(f"**{j}. {source_type}** [Relevance: {source['score']:.3f}]")
                                        st.caption(source['preview'])

    def show_quick_questions(self):
        """Display quick question buttons"""
        st.subheader("🚀 Quick Start")

        # Predefined questions
        quick_questions = [
            "What is IoT security?",
            "What are the vulnerabilities of Qualcomm Snapdragon?",
            "How to protect IoT devices from attacks?",
            "Analyze the latest trends in IoT vulnerabilities",
            "Best Practices for Industrial IoT Security",
            "IoT device authentication and encryption methods"
        ]

        # Create button grid
        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(question, key=f"quick_{i}"):
                    self.process_user_input(question)

    def show_input_area(self):
        """Display input area"""
        # Create input form
        with st.form("chat_form", clear_on_submit=True):
            # Text input
            user_input = st.text_area(
                "💭 Please enter your question:",
                height=100,
                placeholder="Example: What are the vulnerabilities of dglogik inc dglux server?...",
                help="Supports both Chinese and English queries, the more detailed the description, the better"
            )

            # Submit button
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                submitted = st.form_submit_button("🚀 Send", use_container_width=True)

            # Process input
            if submitted and user_input.strip():
                self.process_user_input(user_input.strip())

    def process_user_input(self, user_input: str):
        """Process user input"""
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(user_message)

        # Show processing status
        with st.spinner("🤔 Thinking and analyzing..."):
            try:
                # Call existing Agent (completely unmodified)
                response = st.session_state.agent.process_query(user_input)

                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.now(),
                    "metadata": {
                        "task_type": response.task_type,
                        "confidence": response.confidence,
                        "processing_time": response.processing_time,
                        "model_used": response.model_used,
                        "sources": response.sources,
                        "tokens_used": response.metadata.get("tokens_used", 0)
                    }
                }

                st.session_state.messages.append(assistant_message)

                # Show success message
                st.success(f"✅ Response completed! Confidence: {response.confidence:.1%}")

            except Exception as e:
                # Error handling
                error_message = {
                    "role": "assistant",
                    "content": f"😟 Sorry, an error occurred while processing your question: {str(e)}\n\nPlease check:\n1. Whether Ollama service is running normally\n2. Whether network connection is normal\n3. Whether the model is loaded correctly",
                    "timestamp": datetime.now(),
                    "metadata": {
                        "task_type": "error",
                        "confidence": 0.0,
                        "processing_time": 0.0,
                        "error": str(e)
                    }
                }

                st.session_state.messages.append(error_message)
                st.error(f"❌ Processing failed: {e}")

        # Rerun to display new messages
        st.rerun()

    def export_conversation(self):
        """Export conversation record"""
        if not st.session_state.messages:
            st.warning("No conversation record to export")
            return

        # Prepare export data
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_messages": len(st.session_state.messages),
            "agent_config": st.session_state.agent_config,
            "conversation": []
        }

        for msg in st.session_state.messages:
            export_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"].isoformat(),
                "metadata": msg.get("metadata", {})
            }
            export_data["conversation"].append(export_msg)

        # Generate JSON file
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

        # Provide download
        st.download_button(
            label="📥 Download Conversation Record (JSON)",
            data=json_str,
            file_name=f"iot_security_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def show_footer(self):
        """Display footer"""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                🛡️ IoT Security Intelligent Assistant | Based on RAG+Ollama Technology | 
                <a href='#' style='color: #666;'>Usage Guide</a> | 
                <a href='#' style='color: #666;'>Feedback</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    def run(self):
        """Run Streamlit application"""
        # Create sidebar
        self.create_sidebar()

        # Display main chat interface
        self.show_chat_interface()

        # Display footer
        self.show_footer()


def main():
    """Main function"""
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        border-color: #ff6b6b;
        color: #ff6b6b;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .metric-container {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create and run Web interface
    app = StreamlitWebInterface()
    app.run()


if __name__ == "__main__":
    main()