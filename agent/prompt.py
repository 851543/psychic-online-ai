from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示（system prompt）。"""
    system_role = request.runtime.context.system_role
    base_prompt = "你是一个乐于助人的助手。"

    if system_role == "psychic-online":
        return "你是通灵小助手，可以回答通灵在线教育系统的信息，也可以查询用户通灵在线教育系统的详细信息"

    return base_prompt
