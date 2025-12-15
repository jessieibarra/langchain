from dotenv import load_dotenv
import uuid
from typing import Literal, TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.graph import END, START, StateGraph
from pprint import pprint

load_dotenv()


# Define State schemas

class ChatClassification(TypedDict):
    intent: Literal["ask_question", "request_playlist", "unknown", "greeting", "explore"]
    confidence: float
    signals: dict
    
class DJState(TypedDict):
    user_input: str

    # Understand user intent
    classification: ChatClassification

    # DJ decision
    action: str
    action_reason: str

    # Output
    response_type: str
    response: str
    expects_followup: bool


# Define Nodes

def classify_intent(state: DJState) -> DJState:
    """Use LLM to classify user's intent, then route accordingly"""

    structured_llm = llm.with_structured_output(ChatClassification)

    classification_prompt = f"""
    You are classifying a chat message for a DJ assistant.
    Input: {state["user_input"]}
    Classify the user's intent.
    If the intent is unclear, return intent="unknown" with low confidence.
    Return intent, confidence (0.0–1.0), and any relevant signals.
    """
    classification = structured_llm.invoke(classification_prompt)

    return {"classification": classification}


def decide_action(state: DJState) -> DJState:
    """
    Based on the user's intent, decide whether to 'chat', 'generate_playlist', 
    or 'clarify' and give the reason so we can debug in LangSmith
    """

    classification = state["classification"]

    intent = classification["intent"]
    confidence = classification["confidence"]

    # Default action
    action = "chat"
    reason = "default conversational response"

    if intent == "request_playlist" and confidence >= 0.7:
        action = "generate_playlist"
        reason = "user clearly requested a playlist"

    elif intent in ("explore", "request_playlist") and confidence < 0.7:
        action = "clarify"
        reason = "more information is needed to give the user a quality playlist"

    elif intent == "greeting":
        action = "chat"
        reason = "user is greeting or opening conversation"

    elif intent == "ask_question":
        action = "chat"
        reason = "user asked a general question"

    return {
        "action": action,
        "action_reason": reason,
    }


def generate_response(state: DJState) -> DJState:
    action = state["action"]
    classification = state["classification"]

    # Default values
    response_type = "chat"
    expects_followup = True

    # Chat
    if action == "chat":
        prompt = f"""
        You are a friendly underground DJ.
        The user said: {state["user_input"]}
        Respond casually and warmly. Keep it short.
        Invite them to talk about music if appropriate.
        """

        response = llm.invoke(prompt)

        return {
            "response_type": "chat",
            "response": response.content,
            "expects_followup": True,
        }

    # Clarify
    if action == "clarify":
        prompt = f"""
        You are an underground DJ helping someone figure out what they want.
        The user seems unsure about their music taste.
        Ask ONE clear, simple question to help narrow things down.
        Avoid listing too many options.
        """

        response = llm.invoke(prompt)

        return {
            "response_type": "clarify",
            "response": response.content,
            "expects_followup": True,
        }

    # Generate Playlist
    if action == "generate_playlist":
        prompt = f"""
        You are an underground DJ.
        Create a short playlist inspired by the user's intent.
        Rules:
        - Prefer deep cuts
        - Avoid radio hits
        - 3 tracks max
        - Format: Artist – Track
        - No explanations
        User input: {state["user_input"]}
        """

        response = llm.invoke(prompt)

        return {
            "response_type": "playlist_proposal",
            "response": response.content,
            "expects_followup": True,
        }

    # --- FALLBACK ---
    return {
        "response_type": "chat",
        "response": "Tell me a bit more about what you’re in the mood for.",
        "expects_followup": True,
    }



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)


# Build the graph
builder = StateGraph(DJState)

# Add Nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("decide_action", decide_action)
builder.add_node("generate_response", generate_response)


# Add Edges
builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "decide_action")
builder.add_edge("decide_action", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

