import os
import sys
import time

from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer

from agent.prompt import user_role_prompt
from agent.tool import search, get_weather, handle_tool_errors, CustomContext, update_user_info, greet, CustomState
from langchain.agents.middleware import SummarizationMiddleware

os.environ['OLLAMA_HOST'] = "http://localhost:11434"

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=1,
    streaming=True
)

agent = create_agent(
    llm,
    # tools=[search, get_weather],
    # middleware=[handle_tool_errors, SummarizationMiddleware(
    #     llm,
    #     trigger=("tokens", 4000),
    #     keep=("messages", 20)
    # )],
    # state_schema=CustomState,
    # context_schema=CustomContext,
)

# for chunk in agent.stream(
#         {"messages": [{"role": "user", "content": "旧金山的天气如何？"}]},
#         context=CustomContext(user_id="user_123", user_role="expert"),
#         stream_mode="updates",
# ):
#     for step, data in chunk.items():
#         print(f"步骤: {step}")
#         print(f"内容: {data}")

for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        # context=CustomContext(user_id="user_123", system_role="None"),
        stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")

