from langchain_core.runnables import Runnable, RunnableConfig


class FinAssistant:
    """图里的一个”助理节点“工厂"""
    def __init__(self,runnable:Runnable):
        """
        初始化助理节点。

        :param runnable: prompt 与工具绑定后得到的 Runnable。
        """
        self.runnable = runnable
    def __call__(self,state:dict,config:RunnableConfig):
        """
        执行本节点。

        这里套一层重试：大模型偶尔会返回空内容且不带工具调用，
        这种"空转"会让图卡住。补一句提示让它重新输出，而不是直接崩。

        :param state: 当前图状态。
        :param config: 运行配置，里面有 thread_id / user_id。
        :return: 只包含 messages 的状态增量。
        """
        while True:
            result = self.runnable.invoke(state)
            #既没有调用工具，内容又是空，视为无效输出，重新问一次
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content,list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user","请提供一个真实的输出作为回应")]
                state = {**state,"messages":messages}
            else:
                break
        return {"messages":result}