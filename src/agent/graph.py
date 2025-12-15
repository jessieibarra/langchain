from dotenv import load_dotenv
import uuid
from typing import Literal, TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.graph import END, START, StateGraph
from pprint import pprint

load_dotenv()

# Define state schemas
class DJState(TypedDict):
    taste: str  # raw user input
    interpreted_taste: dict  
    vibe: str   # DJ intent
    curation_rules: dict
    playlist: str


# Define Nodes/Edges
def understand_taste(state: DJState) -> DJState:
    taste = state["taste"].lower()

    interpreted = {
        "mood": "chill" if "chill" in taste else "neutral",
        "energy": "low" if "chill" in taste else "medium"
    }
        
    return {"interpreted_taste": interpreted}

def apply_dj_guidelines(state: DJState) -> DJState:
    taste = state["taste"].lower()

    # Default underground posture
    vibe = "deep"
    rules = {
        "avoid_radio_hits": True,
        "prefer_deep_cuts": True,
        "allow_live_tracks": True,
        "allow_remixes": False,
        "tone": "confident, understated, underground"
    }
    
    # Light adjustment based on taste
    if any(word in taste for word in ["party", "dance", "upbeat"]):
        vibe = "groovy"
        rules.update({
            "allow_remixes": True,
            "energy": "medium-high"
        })

    if any(word in taste for word in ["ambient", "focus", "chill"]):
        vibe = "minimal"
        rules.update({
            "energy": "low",
            "allow_live_tracks": False,
            "prefer_instrumental": True,
        })

    return {
        "vibe": vibe,
        "curation_rules": rules,
    }

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

def generate_playlist(state: DJState) -> DJState:
    system_prompt = f"""
You are an underground DJ.

Branding guidelines:
- Tone: {state["curation_rules"]["tone"]}
- Prefer deep cuts, B-sides, non-radio tracks
- Avoid mainstream hits
- Vibe: {state["vibe"]}

Rules:
{state["curation_rules"]}

Provide a short playlist (3 tracks max).
Do not explain your choices.
Just list artist – track.
"""

    user_prompt = f"""
User taste or mood:
{state["taste"]}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {"playlist": response.content}

# Build the graph
builder = StateGraph(DJState)

# Add Nodes
builder.add_node("understand_taste", understand_taste)
builder.add_node("apply_dj_guidelines", apply_dj_guidelines)
builder.add_node("generate_playlist", generate_playlist)


# Add Edges
builder.add_edge(START, "understand_taste")
builder.add_edge("understand_taste", "apply_dj_guidelines")
builder.add_edge("apply_dj_guidelines", "generate_playlist")
builder.add_edge("generate_playlist", END)

graph = builder.compile()

