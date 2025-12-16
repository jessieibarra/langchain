"""
DJ Agent Evaluation: conversation tone (LLM-as-judge)
"""

from langsmith import evaluate
from openai import OpenAI
from agent.graph import graph

client = OpenAI()


def target(inputs: dict) -> dict:
    """Run the graph with dataset inputs."""
    return graph.invoke(inputs)


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
        evaluators=[conversation_tone],
        experiment_prefix="dj-tone",
    )
    print(results)

