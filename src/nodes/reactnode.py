"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List,Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_react_agent


# wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains the node functions for RAG workflow"""

    def __init__(self,retriver,llm):
        self.retriver = retriver
        self.llm = llm
        self._agent = None   # lazy init agent

    def retrive_docs(self,state:RAGState) -> RAGState:
        """Classic retriver node """
        docs = self.retriver.invoke(state.question)
        return RAGState(
            question=state.question,
            retrived_docs=docs
        )
    
    # build tools
    def _build_tools(self) -> List[Tool]:
        """Build retriver + wikipedia tools"""

        def retriver_tool_fn(query:str)->str:
            docs : List[Document] = self.retriver.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i , d in enumerate(docs[:8],start=1):
                meta = d.metadata if hasattr(d,"metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        
        retriver_tool = Tool(
            name="retriver",
            description="Fetch passages from indexed vectorstore",
            func = retriver_tool_fn
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3,lang="en")
        )

        wikipedia_tool = Tool(
            name = "wikipedia",
            description="Search Wikipedia for general knowledge",
            func = wiki.run,
        )

        return [retriver_tool,wikipedia_tool]

    # build agent
    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriver' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final user answer."
        )
        self._agent = create_react_agent(self.llm, tools=tools, state_modifier=system_prompt)


    def generate_answer(self,state: RAGState) -> RAGState:
        """
        Generate ansewr using ReAct agent with retirver + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages",[])

        answer: Optional[str] = None

        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg,"content",None)

        return RAGState(
            question=state.question,
            retrived_docs=state.retrived_docs,
            answer= answer or  "Could not generate answer."
        )
