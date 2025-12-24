import asyncio
import json
import os

from fastapi import FastAPI, Form
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from starlette.responses import StreamingResponse

from agent.tool import search, get_weather, handle_tool_errors, CustomContext, update_user_info, greet, CustomState, \
    my_course
from agent.prompt import user_role_prompt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OLLAMA_HOST'] = "http://localhost:11434"

appName = "psychic-online"

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=1,
    streaming=True
)

summarizationMiddleware = SummarizationMiddleware(
    llm,
    trigger=("tokens", 4000),
    keep=("messages", 20)
)

agent = create_agent(
    llm,
    tools=[my_course],
    middleware=[handle_tool_errors, user_role_prompt, summarizationMiddleware],
    state_schema=CustomState,
    context_schema=CustomContext,
)

app = FastAPI(root_path="/ai")


async def stream_tokens(prompt: str, user_id: str):
    try:
        for token, metadata in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                context=CustomContext(user_id=user_id, system_role="psychic-online"),
                stream_mode="messages",
        ):
            if metadata["langgraph_node"] == "tools":
                continue
            elif token.text == "":
                continue
            yield f"data: {json.dumps({'type': 'token', 'content': token.text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/chat")
async def chat(user_id: str = Form(), prompt: str = Form()):
    return StreamingResponse(
        stream_tokens(prompt, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=10095)
