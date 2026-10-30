from langchain_core.messages import AnyMessage, HumanMessage


def get_lat_human_message(messages:list[AnyMessage]) -> HumanMessage:
    #反向遍历列表，找到最后一个HumanMessage实例
    for message in reversed(messages):
        if isinstance(message,HumanMessage):
            return message
    raise ValueError("在messages里面没有找到HumanMessage")