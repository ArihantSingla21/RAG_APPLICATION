import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
from pathlib import Path
from src.components.LLM import get_llm_client
from dotenv import load_dotenv
import time
from src.pipelines.work import RAGPipeline
from src.exception import CustomException

# Load environment variables
load_dotenv()

# Custom CSS for better UI
st.set_page_config(
    page_title="📚 Smart PDF Assistant - Your Intelligent PDF Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main > div {
            padding: 2rem;
            border-radius: 10px;
            background: #f8f9fa;
            margin-bottom: 1rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .success-message {
            padding: 1rem;
            border-radius: 5px;
            background-color: #d4edda;
            color: #155724;
            margin: 1rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 5px;
            background-color: #f8d7da;
            color: #721c24;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "mistral-7b"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

def initialize_llm():
    """Initialize the LLM with error handling"""
    try:
        model_key = st.session_state.get('selected_model', 'mistral-7b')
        llm_client = get_llm_client(
            model_key=model_key,
            temperature=0.7,
            max_new_tokens=512
        )
        return llm_client
    except Exception as e:
        st.error(f"🚫 Error initializing LLM: {str(e)}")
        st.info("💡 Please check your HUGGINGFACE_API_KEY in the .env file.")
        return None

def process_pdf(uploaded_file, progress_bar):
    """Process PDF with progress tracking"""
    try:
        # Update progress
        progress_bar.progress(20)
        
        # Get file content
        pdf_content = uploaded_file.getvalue()
        
        # Initialize model with selected model type
        st.session_state.pipeline.initialize_model(
            model_key=st.session_state.selected_model
        )
        progress_bar.progress(40)
        
        # Process PDF
        success = st.session_state.pipeline.process_pdf(
            pdf_content=pdf_content,
            filename=uploaded_file.name
        )
        progress_bar.progress(100)
        
        return success
    except CustomException as ce:
        st.error(f"🚫 {str(ce)}")
        return False
    except Exception as e:
        st.error(f"🚫 Unexpected error while processing PDF: {str(e)}")
        return False

# Sidebar UI
with st.sidebar:
    st.image("https://raw.githubusercontent.com/your-repo/path-to-logo.png", width=100)  # Add your logo
    st.title("⚙️ Settings")
    
    # Model selection with descriptions
    st.subheader("🤖 Model Selection")
    model_options = ["mistral-7b", "zephyr", "phi-2", "llama2-7b"]
    selected_model = st.selectbox(
        "Choose your AI model",
        model_options,
        index=model_options.index(st.session_state.selected_model),
        help="Select the AI model that best suits your needs"
    )

    # Model information cards
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    model_info = {
        "mistral-7b": {
            "name": "Mistral 7B",
            "description": "Powerful instruction-tuned model with excellent reasoning capabilities.",
            "best_for": "Complex reasoning and detailed analysis"
        },
        "zephyr": {
            "name": "Zephyr 7B",
            "description": "Enhanced model tuned with DPO for better following human preferences.",
            "best_for": "Natural conversations and precise instructions"
        },
        "phi-2": {
            "name": "Phi-2",
            "description": "Microsoft's smaller model with strong reasoning on a compact architecture.",
            "best_for": "Quick responses and efficient processing"
        },
        "llama2-7b": {
            "name": "LLaMa 2 7B",
            "description": "Meta's conversational assistant with extensive pre-training.",
            "best_for": "General-purpose conversations and analysis"
        }
    }

    with st.expander(f"ℹ️ About {model_info[selected_model]['name']}"):
        st.markdown(f"""
        **Description:** {model_info[selected_model]['description']}
        
        **Best for:** {model_info[selected_model]['best_for']}
        """)

# Main UI
st.title("📚 Smart PDF Assistant")
st.markdown("### 🔍 Upload your PDF and get instant answers to your questions!")

# File upload section
uploaded_file = st.file_uploader("📄 Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # File info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.2f} KB"
    }
    st.json(file_details)
    
    # Process PDF button
    if st.button("🚀 Process PDF"):
        progress_bar = st.progress(0)
        with st.spinner("📊 Processing your PDF..."):
            try:
                # Process the PDF using the pipeline
                success = st.session_state.pipeline.process_pdf(
                    pdf_content=uploaded_file.getvalue(),
                    filename=uploaded_file.name
                )
                
                if success:
                    progress_bar.progress(100)
                    st.success("✅ PDF processed successfully!")
                    st.session_state.index = True
                    st.balloons()
                else:
                    st.error("❌ Failed to process PDF")
                    
            except CustomException as ce:
                st.error(f"🚫 {str(ce)}")
                progress_bar.progress(0)
            except Exception as e:
                st.error(f"🚫 Unexpected error: {str(e)}")
                progress_bar.progress(0)

    # Question input - Modified to make it more visible
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 💭 Ask me anything about your PDF!")
        question = st.text_input("Your question:", placeholder="e.g., What are the main topics discussed?")
        
        if question:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.pipeline.query_pdf(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response['response']
                    })
                    
                    # Display chat history
                    for chat in st.session_state.chat_history:
                        with st.container():
                            st.markdown(f"**🙋 Question:** {chat['question']}")
                            st.markdown(f"**🤖 Answer:** {chat['answer']}")
                            st.markdown("---")
                    
                except CustomException as ce:
                    st.error(f"🚫 {str(ce)}")
                except Exception as e:
                    st.error(f"🚫 Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ by Arihant Singla | <a href="https://github.com/ArihantSingla21">GitHub</a></p>
    </div>
""", unsafe_allow_html=True) 