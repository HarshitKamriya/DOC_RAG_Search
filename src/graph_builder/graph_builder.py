"""Graph builder for LangGraph  workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNode

class GraphBuilder:
    """Builds and manages Langgraph workflow"""

    def __init__(self,retriver,llm):
        """
        Initialize graph builder

        Args:
            retriver : Document retriver instance
            llm : language model instance
        """

        self.nodes = RAGNode()
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph

        Returns:
            Compiled graph instance        
        """
        # Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriver",self.nodes.retrive_docs)
        builder.add_node("responder",self.nodes.generate_answer)

        # Set entry point
        builder.set_entry_point("retriver")

        # Add edges 
        builder.add_edge("retriver","responder")
        builder.add_edge("responder",END)

        # Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self,question:str) -> dict:
        """
        Run the RAG workflow

        Args:
            question: User question
        
        Returns:
            Final state with answer
        """

        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)


