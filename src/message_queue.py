import os
import queue
import random
import threading
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from _preload import api_key, base_url

llm = ChatOpenAI(model="gpt-5.2", api_key=api_key, base_url=base_url)

messages: list[BaseMessage] = []

# queue.Queue 是线程安全的，可以在多个线程之间共享数据而不需要额外的锁机制。
input_queue: queue.Queue[str] = queue.Queue()

_EXIT = "/exit"

# PromptSession 替代 input()，配合 patch_stdout 实现：
# 主线程 print 时自动清除输入行 → 打印内容 → 重绘输入行，彻底解决输出乱序问题。
session = PromptSession(">>> ")


def input_reader():
    """独立线程：持续读取用户输入并推入队列，LLM 执行期间输入不会丢失。"""
    while True:
        user_input = session.prompt()
        if user_input == _EXIT:
            # 立即终止整个进程，不等待 LLM 执行完毕
            os._exit(0)
        input_queue.put(user_input)
        if input_queue.qsize() > 1:
            previews = [str(item)[:10] for item in list(input_queue.queue)[1:]]
            print(f"[等待队列 {len(previews)} 条: {' | '.join(previews)}]", flush=True)


# patch_stdout 包裹整个主循环：所有 print 调用都会先清除当前输入行，
# 打印完毕后自动重绘 ">>> " 提示符，无需手动补印。
with patch_stdout():
    # 启动输入线程，主线程继续执行 LLM 调用，输入线程持续接收用户输入并推入队列，LLM 执行期间输入不会丢失。
    input_thread = threading.Thread(target=input_reader, daemon=True)
    input_thread.start()

    while True:
        # 阻塞等待至少一条输入
        user_input = input_queue.get()
        messages.append(HumanMessage(content=user_input))

        # 再把队列里积压的其余消息一并取出（非阻塞）
        # 
        # empty() 和 get_nowait() 两步之间不是原子操作，理论上存在竞态——另一个线程可能在两步
        # 之间把元素取走，导致 get_nowait() 还是抛 Empty。但在这个项目里主线程是唯一的消费者，
        # 不存在竞争，所以是安全的。
        while not input_queue.empty():
            messages.append(HumanMessage(content=input_queue.get_nowait()))

        # 所有积压消息合并后，只调用一次 LLM
        ai_message: AIMessage = llm.invoke(messages)
        time.sleep(random.uniform(0, 5))  # 模拟 LLM 慢速输出
        
        messages.append(ai_message)
        print(f"\nAI: \n{ai_message.content}\n")
