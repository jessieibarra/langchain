from agent.graph import graph
from pprint import pprint

# for update in graph.stream(
#     {"user_input": "Some rage against the machine"},
#     stream_mode="updates",
# ):
#     pprint(update)

initial_state = {
    "user_input": "Do you know what music I like?"
}

config = {"configurable": {"thread_id": "customer_123"}}
result = graph.invoke(initial_state, config)

print(result)
