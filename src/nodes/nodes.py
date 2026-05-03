"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNode:
    """Contains node function for RAG workflow"""

    def __init__(self,retriver,llm):
        """
        Initialize RAG nodes 

        Args: 
            retriver: Document retriver instance
            llm : language model instance

        """
        self.retriver = retriver
        self.llm = llm

    def retrive_docs(self,state:RAGState) -> RAGState:
        """
        Retriver relevant documents node

        Args : 
            state : Current RAG state

        Retruns:
            updated RAG state with retrived documents
        """
        docs = self.retriver.invoke(state.question)
        return RAGState(
            question=state.question,
            retrived_docs=docs
        )
    def generate_answer(self,state : RAGState) -> RAGState:
        """
        Generate answer from retrived documents node

        Args:
            state: Current RAG state with retrived documents

        Returns:
            updated RAG state with generate answer.
        """
        # Combine retrived documents into contenxt 
        context = "\n\n".join([doc.page_content for doc in state.retrived_docs])

        # Create prompt
        prompt = f"""Answer the question based on the context.
                context:
                {context}

                Question: {state.question}"""
        
        # Generate response
        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrived_docs=state.retrived_docs,
            answer = response.content
        )
        