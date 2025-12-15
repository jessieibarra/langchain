from agent.graph import graph
from pprint import pprint

for update in graph.stream(
    {"taste": "Some rage against the machine"},
    stream_mode="updates",
):
    pprint(update)
