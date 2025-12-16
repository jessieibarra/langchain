"""
DJ Agent Evaluation: Combined metrics (playlist quality + conversation tone)
"""

from langsmith import evaluate
from openai import OpenAI
from agent.graph import graph

client = OpenAI()


def target(inputs: dict) -> dict:
    """Run the graph with dataset inputs."""
    return graph.invoke(inputs)


def playlist_quality(inputs: dict, outputs: dict) -> dict:
    """
    LLM-as-judge evaluator for playlist quality.
    
    Judges whether the proposed playlist matches the user's request
    based purely on the conversation and output.
    """
    # Get conversation context
    messages = inputs.get("messages", [])
    conversation = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}" 
        for m in messages
    )
    
    # Get proposed tracks
    tracks = outputs.get("proposed_tracks", [])
    if not tracks:
        return {"key": "playlist_quality", "score": 0, "comment": "No playlist generated"}
    
    track_list = "\n".join(f"- {t['artist']} – {t['title']}" for t in tracks)
    vibe = outputs.get("response", "")
    
    prompt = f"""You are evaluating a DJ's playlist recommendation.

CONVERSATION:
{conversation}

PROPOSED PLAYLIST:
{track_list}

DJ's vibe description: {vibe}

Rate the playlist quality from 0.0 to 1.0 based on:
1. Does it match what the user asked for in the conversation?
2. Do the tracks fit the mood/genre the user requested?
3. Is it a coherent playlist that flows well together?

Respond with ONLY a JSON object:
{{"score": 0.X, "reason": "brief explanation"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    
    import json
    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "playlist_quality",
            "score": float(result.get("score", 0)),
            "comment": result.get("reason", "")
        }
    except:
        return {"key": "playlist_quality", "score": 0, "comment": "Failed to parse judge response"}


def conversation_tone(inputs: dict, outputs: dict) -> dict:
    """
    LLM-as-judge evaluator for DJ conversation tone.
    
    The DJ should be: friendly, knowledgeable, casual, not pushy,
    like an underground DJ friend who knows their stuff.
    """
    # Get the DJ's response
    response = outputs.get("response", "")
    if not response:
        return {"key": "conversation_tone", "score": 0, "comment": "No response"}
    
    prompt = f"""You are evaluating a DJ assistant's conversational tone.

The DJ should sound like a friendly underground DJ friend:
- Knowledgeable about music but not pretentious
- Warm and casual, not corporate or robotic  
- Enthusiastic without being over-the-top
- Helpful without being pushy about playlists

DJ'S RESPONSE:
{response}

Rate the tone from 0.0 to 1.0:
- 1.0 = Sounds like a cool DJ friend
- 0.5 = Generic/neutral assistant tone
- 0.0 = Robotic, corporate, or off-putting

Respond with ONLY a JSON object:
{{"score": 0.X, "reason": "brief explanation"}}"""

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    
    import json
    try:
        parsed = json.loads(result.choices[0].message.content)
        return {
            "key": "conversation_tone",
            "score": float(parsed.get("score", 0)),
            "comment": parsed.get("reason", "")
        }
    except:
        return {"key": "conversation_tone", "score": 0, "comment": "Failed to parse"}


if __name__ == "__main__":
    results = evaluate(
        target,
        data="dj-agent-golden-dataset",
        evaluators=[playlist_quality, conversation_tone],
        experiment_prefix="dj-combined",
        num_repetitions=3,
    )
    print(results)

