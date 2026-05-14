"""arXiv paper search skill."""

from datetime import datetime, timedelta
from typing import Any

import arxiv

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class ArxivFetcherSkill(BaseSkill):
    """Fetch recent papers from arXiv."""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="arxiv_fetcher",
            description="Search and fetch papers from arXiv.org by category, keywords, or date range.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(
                    name="query",
                    description="Search query (keywords, title, abstract)",
                    type="string",
                    required=False,
                    default="",
                ),
                SkillParameter(
                    name="categories",
                    description="arXiv categories, e.g., ['hep-th', 'quant-ph']",
                    type="list",
                    required=False,
                    default=["hep-th", "hep-ph", "quant-ph", "gr-qc", "astro-ph.CO"],
                ),
                SkillParameter(
                    name="max_results",
                    description="Maximum number of results to return",
                    type="integer",
                    required=False,
                    default=10,
                ),
                SkillParameter(
                    name="days_back",
                    description="Only fetch papers from the last N days",
                    type="integer",
                    required=False,
                    default=7,
                ),
                SkillParameter(
                    name="sort_by",
                    description="Sort order: 'relevance', 'lastUpdatedDate', 'submittedDate'",
                    type="string",
                    required=False,
                    default="submittedDate",
                ),
            ],
            tags=["academic", "paper", "arxiv", "research"],
            instructions="""
Use this skill to search for academic papers on arXiv.
You can filter by physics categories like hep-th, quant-ph, gr-qc, etc.
The results include paper title, authors, abstract, arXiv ID, and PDF URL.
This skill is read-only and does not modify any data.
            """,
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = parameters.get("query", "")
        categories = parameters.get("categories", ["hep-th", "hep-ph", "quant-ph", "gr-qc", "astro-ph.CO"])
        max_results = parameters.get("max_results", 10)
        days_back = parameters.get("days_back", 7)
        sort_by = parameters.get("sort_by", "submittedDate")

        search_query = query
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({cat_query})" if search_query else cat_query

        if days_back > 0:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            date_str = cutoff_date.strftime("%Y%m%d")
            date_filter = f"submittedDate:[{date_str}000000 TO 999999999999]"
            search_query = f"({search_query}) AND {date_filter}" if search_query else date_filter

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)

        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query or "all",
                max_results=max_results,
                sort_by=sort_criterion,
            )

            papers = []
            for result in client.results(search):
                papers.append(
                    {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "arxiv_id": result.entry_id.split("/")[-1],
                        "pdf_url": result.pdf_url,
                        "published": result.published.isoformat(),
                        "updated": result.updated.isoformat(),
                        "categories": result.categories,
                        "comment": result.comment or "",
                    }
                )

            return {
                "success": True,
                "result": {
                    "papers": papers,
                    "total": len(papers),
                    "query_used": search_query,
                },
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"arXiv API error: {str(e)}",
            }
