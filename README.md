# DJ Agent

A LangGraph-based DJ assistant that helps users discover music and create playlists.

## Graph Overview

The graph uses a **routing pattern** to classify user intent and route to appropriate handlers:

1. **Classify Intent**: LLM classifies user messages (greeting, question, explore, request_playlist)
2. **Decide Action**: Routes based on intent and confidence to:
   - **Chat**: General conversation about music
   - **Clarify**: Ask questions to understand preferences
   - **Generate Playlist**: Create playlist proposal → confirm → search Spotify → create playlist

<div align="center">
  <img src="./graph.png" alt="DJ Agent Graph" width="75%" />
</div>

## Getting Started

1. Install dependencies, along with the [LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/), which will be used to run the server.

```bash
cd path/to/your/app
pip install -e . "langgraph-cli[inmem]"
```

2. (Optional) Customize the code and project as needed. Create a `.env` file if you need to use secrets.

```bash
cp .env.example .env
```

If you want to enable LangSmith tracing, add your LangSmith API key to the `.env` file.

```text
# .env
LANGSMITH_API_KEY=lsv2...
```

3. Start the LangGraph Server.

```shell
langgraph dev
```

For more information on getting started with LangGraph Server, [see here](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/).

## How to customize

1. **Define runtime context**: Modify the `Context` class in the `graph.py` file to expose the arguments you want to configure per assistant. For example, in a chatbot application you may want to define a dynamic system prompt or LLM to use. For more information on runtime context in LangGraph, [see here](https://langchain-ai.github.io/langgraph/agents/context/?h=context#static-runtime-context).

2. **Extend the graph**: The core logic of the application is defined in [graph.py](./src/agent/graph.py). You can modify this file to add new nodes, edges, or change the flow of information.

## Development

While iterating on your graph in LangGraph Studio, you can edit past state and rerun your app from previous states to debug specific nodes. Local changes will be automatically applied via hot reload.

Follow-up requests extend the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

For more advanced features and examples, refer to the [LangGraph documentation](https://langchain-ai.github.io/langgraph/). These resources can help you adapt this template for your specific use case and build more sophisticated conversational agents.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates, allowing you to analyze and optimize your chatbot's performance.
