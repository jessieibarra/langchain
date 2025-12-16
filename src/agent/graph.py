from dotenv import load_dotenv
from typing import Literal, TypedDict, Optional, List, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langgraph.graph import END, START, StateGraph
# Note: When using LangGraph Platform (langgraph dev), checkpointing is handled automatically
from operator import add

load_dotenv()


# =============================================================================
# LLM Configuration
# =============================================================================

MODEL_NAME = "gpt-4o-mini"

llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)

# LangSmith metadata config - reused across all LLM calls
LLM_CONFIG = {"metadata": {"model": MODEL_NAME}}


# =============================================================================
# State Schemas
# =============================================================================

class ChatClassification(TypedDict):
    intent: Literal["ask_question", "request_playlist", "unknown", "greeting", "explore"]
    confidence: float
    signals: dict


class Track(TypedDict):
    artist: str
    title: str
    spotify_uri: Optional[str]


class DJState(TypedDict, total=False):
    # Input
    user_input: str
    messages: Annotated[List[BaseMessage], add]  # Conversation history

    # Classification
    classification: ChatClassification

    # Decision
    action: Literal["chat", "clarify", "generate_playlist"]
    action_reason: str

    # Playlist
    proposed_tracks: List[Track]
    user_confirmed: Optional[bool]
    spotify_playlist_url: Optional[str]

    # Response
    response_type: Literal["chat", "clarify", "playlist_proposal", "playlist_created"]
    response: str
    expects_followup: bool


# =============================================================================
# Node: Classify Intent
# =============================================================================

def classify_intent(state: DJState) -> dict:
    """Use LLM to classify user's intent"""

    structured_llm = llm.with_structured_output(ChatClassification)

    messages = [
        SystemMessage(content="""You are classifying chat messages for a DJ assistant.
Classify the user's intent into one of these categories:
- greeting: User is saying hello or opening the conversation
- ask_question: User is asking a general question about music
- explore: User wants to explore music but hasn't decided what
- request_playlist: User is clearly asking for a playlist
- unknown: Intent is unclear

Return intent, confidence (0.0–1.0), and any relevant signals (mood, genre, etc.)."""),
        HumanMessage(content=state["user_input"]),
    ]

    classification = structured_llm.invoke(messages, config=LLM_CONFIG)

    return {
        "classification": classification,
        "messages": [HumanMessage(content=state["user_input"])],
    }


# =============================================================================
# Node: Decide Action
# =============================================================================

def decide_action(state: DJState) -> dict:
    """Route based on classification to chat, clarify, or generate playlist"""

    classification = state["classification"]
    intent = classification["intent"]
    confidence = classification["confidence"]

    # High confidence playlist request → generate playlist
    if intent == "request_playlist" and confidence >= 0.7:
        return {
            "action": "generate_playlist",
            "action_reason": "User clearly requested a playlist",
        }

    # Low confidence or exploring → clarify
    if intent in ("explore", "request_playlist") and confidence < 0.7:
        return {
            "action": "clarify",
            "action_reason": "Need more information for a quality playlist",
        }

    # Everything else → chat
    return {
        "action": "chat",
        "action_reason": f"Handling {intent} with conversational response",
    }


# =============================================================================
# Routing Function for Conditional Edges
# =============================================================================

def route_by_action(state: DJState) -> str:
    """Route to the appropriate node based on action"""
    return state["action"]


# =============================================================================
# Node: Chat Response
# =============================================================================

def handle_chat(state: DJState) -> dict:
    """Handle general chat/greeting/questions"""

    messages = [
        SystemMessage(content="""You are a friendly underground DJ with deep knowledge of music.
Keep responses short and warm. If appropriate, invite the user to talk about music.
Don't be pushy about playlists - just be a good conversationalist about music."""),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(messages, config=LLM_CONFIG)

    return {
        "response_type": "chat",
        "response": response.content,
        "expects_followup": True,
        "messages": [AIMessage(content=response.content)],
    }


# =============================================================================
# Node: Clarify Preferences
# =============================================================================

def handle_clarify(state: DJState) -> dict:
    """Ask clarifying questions to understand user's music taste"""

    signals = state["classification"].get("signals", {})

    messages = [
        SystemMessage(content=f"""You are an underground DJ helping someone discover what they want.
The user seems interested in music but hasn't given enough detail.
Known signals: {signals}

Ask ONE clear, simple question to narrow things down.
Examples of good questions:
- "What's the vibe - something to dance to or something to chill with?"
- "Any artists you've been into lately?"
- "What are you doing while listening - working, driving, party?"

Keep it casual and short. Don't list too many options."""),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(messages, config=LLM_CONFIG)

    return {
        "response_type": "clarify",
        "response": response.content,
        "expects_followup": True,
        "messages": [AIMessage(content=response.content)],
    }


# =============================================================================
# Node: Generate Playlist Proposal
# =============================================================================

class PlaylistProposal(TypedDict):
    tracks: List[Track]
    vibe_description: str


def handle_generate_playlist(state: DJState) -> dict:
    """Generate a playlist proposal based on user input"""

    structured_llm = llm.with_structured_output(PlaylistProposal)

    messages = [
        SystemMessage(content="""You are an underground DJ creating a playlist.

Rules:
- 3 tracks exactly
- Prefer deep cuts over radio hits
- Include a mix that flows well together
- Each track needs: artist, title (no spotify_uri yet)

Also provide a short "vibe_description" (one sentence) capturing the mood."""),
        HumanMessage(content=f"Create a playlist for: {state['user_input']}"),
    ]

    proposal = structured_llm.invoke(messages, config=LLM_CONFIG)

    # Format the response for display
    track_list = "\n".join(
        f"  {i+1}. {t['artist']} – {t['title']}"
        for i, t in enumerate(proposal["tracks"])
    )
    response_text = f"""Here's what I've got for you:

{track_list}

Vibe: {proposal["vibe_description"]}

Want me to create this playlist on Spotify?"""

    return {
        "proposed_tracks": proposal["tracks"],
        "response_type": "playlist_proposal",
        "response": response_text,
        "expects_followup": True,
        "messages": [AIMessage(content=response_text)],
    }


# =============================================================================
# Node: Confirm Playlist (Human-in-the-loop)
# =============================================================================

def confirm_playlist(state: DJState) -> dict:
    """Interrupt to get user confirmation before creating Spotify playlist"""

    tracks = state.get("proposed_tracks", [])
    track_list = "\n".join(f"  - {t['artist']} – {t['title']}" for t in tracks)

    # This will pause execution and wait for user input
    user_response = interrupt(
        f"Ready to create this playlist on Spotify:\n{track_list}\n\nConfirm? (yes/no)"
    )

    # User's response comes back here after they resume
    confirmed = user_response.lower() in ("yes", "y", "yeah", "sure", "do it", "create it")

    return {
        "user_confirmed": confirmed,
        "messages": [HumanMessage(content=user_response)],
    }


# =============================================================================
# Routing: After Confirmation
# =============================================================================

def route_after_confirmation(state: DJState) -> str:
    """Route based on user confirmation"""
    if state.get("user_confirmed"):
        return "search_spotify"
    return "playlist_declined"


# =============================================================================
# Node: Search Spotify
# =============================================================================

def search_spotify(state: DJState) -> dict:
    """Search Spotify for the proposed tracks and get URIs"""

    tracks = state.get("proposed_tracks", [])

    # TODO: Implement actual Spotify API search
    # For now, simulate finding tracks
    found_tracks = []
    for track in tracks:
        found_tracks.append({
            **track,
            "spotify_uri": f"spotify:track:{track['artist'][:3]}{track['title'][:3]}".lower().replace(" ", ""),
        })

    return {
        "proposed_tracks": found_tracks,
    }


# =============================================================================
# Node: Create Spotify Playlist
# =============================================================================

def create_spotify_playlist(state: DJState) -> dict:
    """Create the playlist on Spotify"""

    tracks = state.get("proposed_tracks", [])

    # TODO: Implement actual Spotify API playlist creation
    # For now, simulate creating the playlist
    playlist_url = "https://open.spotify.com/playlist/simulated123"

    track_list = "\n".join(f"  - {t['artist']} – {t['title']}" for t in tracks)
    response_text = f"""Done! Your playlist is ready:
{playlist_url}

Tracks:
{track_list}

Enjoy the tunes! Let me know if you want another one."""

    return {
        "spotify_playlist_url": playlist_url,
        "response_type": "playlist_created",
        "response": response_text,
        "expects_followup": True,
        "messages": [AIMessage(content=response_text)],
    }


# =============================================================================
# Node: Playlist Declined
# =============================================================================

def handle_playlist_declined(state: DJState) -> dict:
    """Handle when user declines the playlist"""

    response_text = "No worries! Want me to try something different, or should we explore other vibes?"

    return {
        "response_type": "chat",
        "response": response_text,
        "expects_followup": True,
        "messages": [AIMessage(content=response_text)],
    }


# =============================================================================
# Build the Graph
# =============================================================================

builder = StateGraph(DJState)

# Add Nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("decide_action", decide_action)
builder.add_node("chat", handle_chat)
builder.add_node("clarify", handle_clarify)
builder.add_node("generate_playlist", handle_generate_playlist)
builder.add_node("confirm_playlist", confirm_playlist)
builder.add_node("search_spotify", search_spotify)
builder.add_node("create_spotify_playlist", create_spotify_playlist)
builder.add_node("playlist_declined", handle_playlist_declined)

# Add Edges
builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "decide_action")

# Conditional routing based on action
builder.add_conditional_edges(
    "decide_action",
    route_by_action,
    {
        "chat": "chat",
        "clarify": "clarify",
        "generate_playlist": "generate_playlist",
    }
)

# Chat and clarify go to END (await next user input)
builder.add_edge("chat", END)
builder.add_edge("clarify", END)

# Playlist flow: propose → confirm → search → create
builder.add_edge("generate_playlist", "confirm_playlist")

# After confirmation, route based on user's response
builder.add_conditional_edges(
    "confirm_playlist",
    route_after_confirmation,
    {
        "search_spotify": "search_spotify",
        "playlist_declined": "playlist_declined",
    }
)

builder.add_edge("search_spotify", "create_spotify_playlist")
builder.add_edge("create_spotify_playlist", END)
builder.add_edge("playlist_declined", END)

# Compile the graph
# Note: LangGraph Platform handles checkpointing automatically
graph = builder.compile()
