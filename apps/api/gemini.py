import os

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


def generate_zone_summary(zone_data: dict) -> str:
    prompt = f"""
    You are writing for a city planner reviewing urban heat data.
    Write 2-3 sentences explaining this zone in plain English.

    Zone severity: {zone_data['severity']}
    Mean heat above city median: {zone_data['mean_relative_heat']:.1f}°C
    Primary contributors: {', '.join(zone_data['top_contributors'])}
    Recommended interventions: {', '.join(zone_data['top_recommendations'])}

    Be specific. Mention surface conditions and expected cooling impact.
    Do not use jargon. Write as if explaining to a non-technical city official.
    """
    response = _get_client().chat.completions.create(
        model="google/gemini-2.5-flash-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
