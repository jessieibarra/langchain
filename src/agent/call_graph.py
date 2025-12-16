from agent.graph import graph
from pprint import pprint
from langgraph.types import Command

# Thread ID for multi-turn conversation
config = {"configurable": {"thread_id": "dj_session_001"}}

# =============================================================================
# Example 1: Simple greeting (chat flow)
# =============================================================================
print("=" * 60)
print("Example 1: Greeting")
print("=" * 60)

result = graph.invoke(
    {"user_input": "Hey! What's up?"},
    config,
)
print(f"Response: {result['response']}\n")

# =============================================================================
# Example 2: Clarify flow
# =============================================================================
print("=" * 60)
print("Example 2: Vague request (clarify flow)")
print("=" * 60)

result = graph.invoke(
    {"user_input": "I want some cool music"},
    config,
)
print(f"Response: {result['response']}\n")

# =============================================================================
# Example 3: Playlist request (will interrupt for confirmation)
# =============================================================================
print("=" * 60)
print("Example 3: Playlist request (with interrupt)")
print("=" * 60)

# First invoke will stop at the interrupt
result = graph.invoke(
    {"user_input": "Make me a chill lo-fi hip hop playlist for studying"},
    config,
)

print(f"Proposed playlist:\n{result['response']}\n")

# Check if we're interrupted
state = graph.get_state(config)
if state.next:
    print(f"Graph paused at: {state.next}")
    print("Waiting for user confirmation...")

    # Resume with user's response (simulating "yes")
    result = graph.invoke(
        Command(resume="yes"),  # User confirms
        config,
    )
    print(f"\nFinal response:\n{result['response']}")
