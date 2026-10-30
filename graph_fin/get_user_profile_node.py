from langchain_core.runnables import RunnableConfig

from graph_fin.state import State
from tools import fin_db


def get_user_profile(state:State,config:RunnableConfig):
    """
    进图第一站：查一次用户画像并写进 State。
    为什么和对话历史分开：风险等级是**结构化事实**，每轮重新查库写入，
    不让模型从历史里"回忆"——回忆错了就是适当性事故。
    """
    user_id = (config or {}).get("configurable",{}).get("user_id","")
    acc = fin_db.get_account(user_id)
    if not acc:
        profile = f"未识别用户（user_id={user_id}）,风险等级未知，禁止发起交易"
    else:
        total = fin_db.get_total_asset(user_id)
        profile = (f"用户 {acc['name']}（{user_id}），风险等级 {acc['risk_level']}，"
                   f"可用资金 {acc['cash']:.2f} 元，总资产约 {total:.2f} 元，"
                   f"单日申购限额 {fin_db.DAILY_LIMIT.get(acc['risk_level'], 0):.0f} 元。")
    return {"user_profile":profile}

