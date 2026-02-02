import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self, 
        persist_dir: str = "faiss_store", 
        embedding_model: str = "all-MiniLM-L6-v2", 
        llm_model: str = "llama-3.1-8b-instant"  # Updated to active model
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables!")
        
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def load_vectorstore(self):
        """Load existing vectorstore if available"""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()
            return True
        return False

    def build_vectorstore(self, documents):
        """Build vectorstore from documents"""
        self.vectorstore.build_from_documents(documents)

    def search_and_answer(self, query: str, top_k: int = 1) -> str:
        """
        Search vectorstore and generate answer using LLM.
        Now uses only the top 1 most relevant chunk for better precision.
        """
        results = self.vectorstore.query(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found in the uploaded files."
        
        # Get the best match
        best_match = results[0]
        context = best_match["metadata"].get("text", "") if best_match["metadata"] else ""
        similarity_score = best_match.get("distance", None)
        
        if not context:
            return "No relevant content found in the uploaded files."
        
        # Create prompt for LLM with only the best matching context
        prompt = f"""Based on the following context from the uploaded documents, answer the question accurately and concisely.

Question: {query}

Context:
{context}

Answer:"""
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        
        # Optionally log similarity score
        print(f"[INFO] Using best match with similarity score (distance): {similarity_score}")
        
        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    rag_search.load_vectorstore()
    query = "What is attention mechanism?"
    answer = rag_search.search_and_answer(query, top_k=1)
    print("Answer:", answer)
