import streamlit as st
import os
from pathlib import Path
import time
import sys
from pathlib import Path
from src.data_loader import load_uploaded_files
from src.search import RAGSearch


# Page configuration
st.set_page_config(
    page_title="Q&A with Local Files",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = False
if 'uploaded_filenames' not in st.session_state:
    st.session_state.uploaded_filenames = []
if 'rag_search' not in st.session_state:
    st.session_state.rag_search = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("📄 Q&A with Local Files")
st.markdown("Upload your documents and ask questions about their content using AI-powered search.")
st.markdown("---")

# File upload section
st.subheader("📁 Upload Your Documents")
uploaded_files = st.file_uploader(
    "Choose files (PDF, TXT, CSV, Excel, Word, JSON)",
    type=['pdf', 'txt', 'csv', 'xlsx', 'xls', 'docx', 'json'],
    accept_multiple_files=True,
    help="Upload one or more documents to create your knowledge base"
)

# Process uploaded files with detailed progress
if uploaded_files:
    # Check if files have changed
    current_filenames = [f.name for f in uploaded_files]
    
    if current_filenames != st.session_state.uploaded_filenames:
        st.session_state.uploaded_filenames = current_filenames
        st.session_state.vectorstore_ready = False
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Processing Documents")
            
            # Progress bar for overall process
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Loading documents
                status_text.markdown("**Step 1/4:** 📂 Loading documents...")
                overall_progress.progress(10)
                
                documents = load_uploaded_files(uploaded_files)
                
                if not documents:
                    st.error("❌ No documents could be loaded. Please check your files.")
                    overall_progress.empty()
                    status_text.empty()
                else:
                    overall_progress.progress(25)
                    status_text.markdown(f"**Step 1/4:** ✅ Loaded {len(documents)} document(s)")
                    time.sleep(0.3)
                    
                    # Step 2: Initializing RAG system
                    status_text.markdown("**Step 2/4:** 🤖 Initializing AI models...")
                    overall_progress.progress(40)
                    
                    st.session_state.rag_search = RAGSearch(
                    persist_dir="streamlit_faiss_store",
                    embedding_model="all-MiniLM-L6-v2",
                    llm_model="llama-3.1-8b-instant"
                    )

                    
                    overall_progress.progress(50)
                    status_text.markdown("**Step 2/4:** ✅ AI models initialized")
                    time.sleep(0.3)
                    
                    # Step 3: Chunking documents
                    status_text.markdown("**Step 3/4:** ✂️ Chunking documents...")
                    overall_progress.progress(60)
                    
                    # Show chunking progress
                    chunking_progress = st.progress(0)
                    chunk_status = st.empty()
                    
                    # Call chunking with progress updates
                    from src.embedding import EmbeddingPipeline
                    emb_pipe = EmbeddingPipeline(
                        model_name="all-MiniLM-L6-v2",
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    
                    chunks = emb_pipe.chunk_documents(documents)
                    chunk_status.markdown(f"Created {len(chunks)} chunks from {len(documents)} documents")
                    chunking_progress.progress(100)
                    
                    overall_progress.progress(70)
                    status_text.markdown(f"**Step 3/4:** ✅ Created {len(chunks)} text chunks")
                    time.sleep(0.3)
                    
                    # Step 4: Generating embeddings
                    status_text.markdown("**Step 4/4:** 🧮 Generating embeddings...")
                    overall_progress.progress(75)
                    
                    # Show embedding progress
                    embedding_progress = st.progress(0)
                    embed_status = st.empty()
                    
                    # Generate embeddings with progress
                    import numpy as np
                    embeddings = emb_pipe.embed_chunks(chunks)
                    embed_status.markdown(f"Generated embeddings for {len(chunks)} chunks")
                    embedding_progress.progress(100)
                    
                    overall_progress.progress(90)
                    
                    # Step 5: Building vector store
                    status_text.markdown("**Step 4/4:** 💾 Building vector store...")
                    
                    # Build vectorstore
                    metadatas = [{"text": chunk.page_content} for chunk in chunks]
                    st.session_state.rag_search.vectorstore.add_embeddings(
                        np.array(embeddings).astype('float32'),
                        metadatas
                    )
                    st.session_state.rag_search.vectorstore.save()
                    
                    overall_progress.progress(100)
                    status_text.markdown("**Step 4/4:** ✅ Vector store built successfully")
                    
                    st.session_state.vectorstore_ready = True
                    
                    # Show success message
                    st.success(f"🎉 Successfully processed {len(documents)} documents into {len(chunks)} searchable chunks!")
                    
                    # Clear progress bars after a moment
                    time.sleep(1)
                    overall_progress.empty()
                    chunking_progress.empty()
                    embedding_progress.empty()
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                st.exception(e)
                st.session_state.vectorstore_ready = False
                overall_progress.empty()
                status_text.empty()
    
    # Display uploaded file names
    if st.session_state.uploaded_filenames:
        st.markdown("**Uploaded Files:**")
        cols = st.columns(3)
        for idx, filename in enumerate(st.session_state.uploaded_filenames):
            col_idx = idx % 3
            with cols[col_idx]:
                st.markdown(f"- 📄 {filename}")

st.markdown("---")

# Query section
st.subheader("💬 Ask Questions")

if st.session_state.vectorstore_ready:
    # Create two columns for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic discussed in these documents?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", use_container_width=True, type="primary")
    
    # Process query with progress indicator
    if ask_button and user_query:
        # Create answer container
        answer_container = st.container()
        
        with answer_container:
            # Show processing steps
            with st.spinner("🔍 Searching through documents..."):
                time.sleep(0.3)  # Brief pause for UX
            
            query_status = st.empty()
            query_status.markdown("🤖 Generating AI-powered answer...")
            
            try:
                # Search and generate answer
                answer = st.session_state.rag_search.search_and_answer(user_query, top_k=5)
                
                # Clear status
                query_status.empty()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": answer
                })
                
                # Force rerun to show updated history
                st.rerun()
                
            except Exception as e:
                query_status.empty()
                st.error(f"❌ Error generating answer: {str(e)}")
                st.exception(e)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📝 Chat History")
        
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                with st.expander(f"Q{len(st.session_state.chat_history) - idx}: {chat['question'][:80]}...", expanded=(idx==0)):
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**💡 Answer:**\n\n{chat['answer']}")
    
else:
    st.info("👆 Please upload files first to start asking questions.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application allows you to:
    - Upload multiple document types
    - Create a searchable knowledge base
    - Ask questions and get AI-powered answers
    
    **Supported formats:**
    - PDF (.pdf)
    - Text (.txt)
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Word (.docx)
    - JSON (.json)
    """)
    
    st.markdown("---")
    
    st.header("🔧 Settings")
    st.markdown(f"""
    **Embedding Model:** all-MiniLM-L6-v2
    
    **LLM Model:** gemma2-9b-it (Groq)
    
    **Chunk Size:** 1000 characters
    
    **Chunk Overlap:** 200 characters
    
    **Top-K Results:** 5
    """)
    
    st.markdown("---")
    
    # System status indicator
    st.header("📊 System Status")
    if st.session_state.vectorstore_ready:
        st.success("✅ Ready for Questions")
        if st.session_state.uploaded_filenames:
            st.metric("Files Loaded", len(st.session_state.uploaded_filenames))
        if st.session_state.chat_history:
            st.metric("Questions Asked", len(st.session_state.chat_history))
    else:
        st.warning("⏳ Waiting for files...")
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Reset system button
    if st.button("🔄 Reset System"):
        st.session_state.vectorstore_ready = False
        st.session_state.uploaded_filenames = []
        st.session_state.rag_search = None
        st.session_state.chat_history = []
        st.rerun()
