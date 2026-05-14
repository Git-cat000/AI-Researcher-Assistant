#!/usr/bin/env python3
"""
AI Researcher Assistant - Basic Example

This example demonstrates how to use the ResearcherAgent to:
1. Search for recent papers on a topic
2. Add papers to the knowledge base
3. Answer research questions with citations
"""

import asyncio
import logging
import os

# Add the project root to the path (if running from examples directory)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_researcher_assistant.core.config import update_config
from ai_researcher_assistant.memory import AcademicRAG
from ai_researcher_assistant.orchestration import ResearcherAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def setup_knowledge_base():
    """Initialize RAG and add some sample papers (optional)"""
    rag = AcademicRAG(collection_name="research_papers")

    # You can pre-load papers here, or let the agent fetch them on demand
    # For demo purposes, we'll add a few placeholder papers
    logger.info("Knowledge base initialized with %d papers", rag.count_papers())
    return rag


async def demo_basic_question():
    """Demo: Answer a simple research question using the agent"""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Research Question")
    print("=" * 60)

    # Create agent with default configuration
    agent = ResearcherAgent(name="Physics Assistant")
    agent.initialize()

    # Ask a question that requires using arXiv fetcher skill
    question = "What are the latest developments in AdS/CFT correspondence? Please find and summarize 2 recent papers."

    print(f"\nUser: {question}\n")
    print("Agent is thinking and may call arXiv...\n")

    answer = await agent.aprocess(question)

    print(f"Assistant: {answer}\n")

    # Show execution stats
    stats = agent.get_stats()
    print(f"Execution stats: {stats.get('steps_taken')} steps taken")
    print(f"Skills available: {stats.get('skills_available')}")

    agent.shutdown()
    return answer


async def demo_with_rag():
    """Demo: Use RAG to answer from pre-loaded knowledge"""
    print("\n" + "=" * 60)
    print("DEMO 2: RAG-based Question Answering")
    print("=" * 60)

    # Create agent with RAG enabled
    agent = ResearcherAgent(name="Physics Librarian", enable_rag=True)
    agent.initialize()

    # First, add a paper to the knowledge base (simulate pre-loading)
    print("Pre-loading a sample paper into knowledge base...")
    agent.add_paper_to_knowledge_base(
        title="Holographic Duality in Condensed Matter Physics",
        abstract="We apply holographic methods to model strange metals and high-Tc superconductors...",
        arxiv_id="2301.12345",
        authors=["J. Maldacena", "S. Hartnoll"],
        categories=["hep-th", "cond-mat.str-el"],
    )

    # Now ask a question that should hit the RAG
    question = "What does the holographic duality paper I just added say about strange metals?"

    print(f"\nUser: {question}\n")
    print("Agent searching knowledge base...\n")

    answer = await agent.aprocess(question)

    print(f"Assistant: {answer}\n")

    agent.shutdown()
    return answer


async def demo_paper_writing():
    """Demo: Use paper writing skill for polishing text"""
    print("\n" + "=" * 60)
    print("DEMO 3: Academic Writing Assistance")
    print("=" * 60)

    agent = ResearcherAgent(name="Writing Assistant")
    agent.initialize()

    # A rough draft
    draft = """
    We did some calculations about black holes and found that information is not lost.
    This is important because it solves a long-standing problem.
    """

    question = f"""Please polish the following text for an academic physics paper:

    {draft}

    Use formal style and proper scientific tone."""

    print(f"\nUser: {question}\n")
    print("Agent polishing text...\n")

    answer = await agent.aprocess(question)

    print(f"Assistant: {answer}\n")

    agent.shutdown()
    return answer


async def main():
    """Run all demos"""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some features may not work.")
        print("Please set your API key in .env file or environment variable.")
        print("You can still run the demos with a mock LLM for testing.")

    # Update configuration if needed
    update_config(data_dir="./data", log_level="INFO")

    try:
        # Run demos
        await demo_basic_question()
        await demo_with_rag()
        await demo_paper_writing()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.exception("Demo failed")
        print(f"\nError: {e}")
        print("Please check your API keys and network connection.")


if __name__ == "__main__":
    asyncio.run(main())
