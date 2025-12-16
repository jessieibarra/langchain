#!/usr/bin/env python3
"""
Simple script to test the DJ Agent graph.
Usage:
    python test_graph.py "Your message here"
    python test_graph.py  # Interactive mode
"""

import sys
import json
from agent.graph import graph


def format_output(result: dict) -> str:
    """Format the graph output for display."""
    output_lines = []
    
    # Show response
    if "response" in result:
        output_lines.append("=" * 60)
        output_lines.append("RESPONSE:")
        output_lines.append("=" * 60)
        output_lines.append(result["response"])
        output_lines.append("")
    
    # Show classification if available
    if "classification" in result:
        classification = result["classification"]
        output_lines.append("=" * 60)
        output_lines.append("CLASSIFICATION:")
        output_lines.append("=" * 60)
        output_lines.append(f"Intent: {classification.get('intent', 'N/A')}")
        output_lines.append(f"Confidence: {classification.get('confidence', 0):.2f}")
        if classification.get("signals"):
            output_lines.append(f"Signals: {classification['signals']}")
        output_lines.append("")
    
    # Show action if available
    if "action" in result:
        output_lines.append("=" * 60)
        output_lines.append("ACTION:")
        output_lines.append("=" * 60)
        output_lines.append(f"Action: {result['action']}")
        output_lines.append(f"Reason: {result.get('action_reason', 'N/A')}")
        output_lines.append("")
    
    # Show playlist if available
    if "proposed_tracks" in result and result["proposed_tracks"]:
        output_lines.append("=" * 60)
        output_lines.append("PROPOSED PLAYLIST:")
        output_lines.append("=" * 60)
        for i, track in enumerate(result["proposed_tracks"], 1):
            artist = track.get("artist", "Unknown")
            title = track.get("title", "Unknown")
            uri = track.get("spotify_uri", "")
            uri_str = f" ({uri})" if uri else ""
            output_lines.append(f"{i}. {artist} – {title}{uri_str}")
        output_lines.append("")
    
    # Show response type
    if "response_type" in result:
        output_lines.append("=" * 60)
        output_lines.append("RESPONSE TYPE:")
        output_lines.append("=" * 60)
        output_lines.append(result["response_type"])
        output_lines.append("")
    
    # Show full state (for debugging)
    if "--debug" in sys.argv or "-d" in sys.argv:
        output_lines.append("=" * 60)
        output_lines.append("FULL STATE (DEBUG):")
        output_lines.append("=" * 60)
        output_lines.append(json.dumps(result, indent=2, default=str))
        output_lines.append("")
    
    return "\n".join(output_lines)


def main():
    """Main function to run the graph."""
    # Get input from command line or prompt
    if len(sys.argv) > 1 and sys.argv[1] not in ["--debug", "-d"]:
        user_input = sys.argv[1]
    else:
        user_input = input("Enter your message: ").strip()
        if not user_input:
            print("No input provided. Exiting.")
            return
    
    # Prepare input for the graph
    inputs = {
        "messages": [
            {"role": "human", "content": user_input}
        ]
    }
    
    print("\n" + "=" * 60)
    print("RUNNING GRAPH...")
    print("=" * 60)
    print(f"Input: {user_input}\n")
    
    try:
        # Invoke the graph
        result = graph.invoke(inputs)
        
        # Format and display output
        output = format_output(result)
        print(output)
        
    except Exception as e:
        print(f"\n❌ Error running graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

