import json
import os

from fastapi import FastAPI, Form, Request
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from starlette.responses import StreamingResponse

from agent.output import ContactInfo
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

agent_review = create_agent(
    llm,
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

agent_code = create_agent(
    llm,
)

app = FastAPI(root_path="/ai")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:88"],  # 或 ["*"]（仅开发）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/course_commit_audit")
async def chat(request: Request):
    raw_body = await request.body()
    json = raw_body.decode("utf-8")
    result = agent_review.invoke({
        "messages": [
            {"role": "user", "content": json}]
    })
    return result["structured_response"].review


@app.post("/answer")
async def chat(request: Request):
    body = await request.body()
    data = json.loads(body)
    result = agent_code.invoke({
        "messages": [
            {"role": "system", "content": "你是一名严谨的编程助教。"},
            {"role": "user", "content": f"""
            请根据题目描述，给出解题思路、关键边界情况，并输出可运行的 {data["language"]} 代码。
            我给了现有代码，请指出问题并给出改进后的完整代码。
            【题目描述】
            {data["problemText"]}
            【语言】{data["language"]}
            {data["code"]}
            """}
        ]
    })
    return result["messages"][2].content


if __name__ == '__main__':
    import uvicorn

uvicorn.run(app, host="localhost", port=10095)
