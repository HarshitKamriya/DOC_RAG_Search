"""Vector stroe module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manges vector store application"""
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriver = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args :
            documents : List of documents to embed
        """

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        self.retriver = self.vectorstore.as_retriever() 

    def get_retriver(self):
        """
        Get the retriver instance

        Returns:
            Retriver instance
        """

        if self.retriver is None:
            raise ValueError("Vector store not initialized. Call create _vectorstore first.")
        return self.retriver
    
    def retrieve(self,query: str,k: int = 4) -> List[Document]:
        """
        Retrives relevant documents for a query

        Args : 
            query : Search query
            k : Number of documents to retrives

        Retruns: 
            List of relevant documents
        """
        if self.retriver is None:
            raise ValueError("Vector store not initialized. Call create_vectorstroe first.")
        
        return self.retriver.invoke(query)
        

