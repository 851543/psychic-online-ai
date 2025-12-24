import requests
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.agents import AgentState
from pydantic import BaseModel
from langgraph.types import Command


class CustomState(AgentState):
    user_id: str  # 用户 ID
    user_name: str  # 用户


class CustomContext(BaseModel):
    user_id: str  # 用户 ID
    system_role: str  # 角色


@tool
def my_course(
        runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """查询我的课程信息"""
    user_id = runtime.context.user_id
    if user_id == 'None':
        return "未登陆系统无法查询"
    res = requests.get("http://localhost:63020/learning/mycoursetable/ai", params={'userId': user_id})
    return res.json()


@tool
def update_user_info(
        runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """查找并更新用户信息。"""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "未知用户"
    return Command(update={
        "user_name": name,
        # 更新消息历史
        "messages": [
            ToolMessage(
                "已成功检索到用户信息",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


@tool
def greet(
        runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """在找到用户信息后使用此工具向用户打招呼。"""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    "请先调用 `update_user_info` 工具来获取并更新用户的姓名。",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"你好 {user_name}!"


@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"Results for: {query}"


@tool
def get_weather(location: str) -> str:
    """获取某个地点的天气信息。"""
    return f"天气 — {location}：晴，72°F"


@wrap_tool_call
def handle_tool_errors(request, handler):
    """在工具执行出错时使用自定义消息进行处理。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义的错误消息
        return ToolMessage(
            content=f"工具错误：请检查你的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
