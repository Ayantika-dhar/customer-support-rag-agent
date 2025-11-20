# utils/search.py

from typing import List, Dict
import os
import sys

from tavily import TavilyClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import get_config


def _get_tavily_client() -> TavilyClient | None:
    """
    Initialize Tavily client from config/env.
    Returns None if no API key is set.
    """
    config = get_config()
    api_key = config.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        # No key configured -> no web search
        return None

    try:
        client = TavilyClient(api_key=api_key)
        return client
    except Exception as e:
        print(f"[web_search] Failed to init Tavily client: {e}")
        return None


def web_search(query: str, k: int = 3) -> List[Dict]:
    """
    Perform a web search and return a normalized list of results:

    [
      {"title": "...", "snippet": "...", "url": "..."},
      ...
    ]
    """
    if not query:
        return []

    client = _get_tavily_client()
    if client is None:
        # Graceful fallback – no crash if key missing
        print("[web_search] No Tavily API key configured.")
        return []

    try:
        response = client.search(
            query=query,
            max_results=k,
            search_depth="basic",  # good enough for our use case
        )
    except Exception as e:
        print(f"[web_search] Tavily search error: {e}")
        return []

    results: List[Dict] = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", "") or "Untitled result",
                "snippet": item.get("content", "")[:400],
                "url": item.get("url", ""),
            }
        )

    return results
