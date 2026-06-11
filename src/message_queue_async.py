import asyncio
import random

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from _preload import api_key, base_url

llm = ChatOpenAI(model="gpt-5.2", api_key=api_key, base_url=base_url)

messages: list[BaseMessage] = []

_EXIT = "/exit"

# patch_stdout 包裹整个事件循环：print 时自动清除输入行、打印后重绘提示符。
# prompt_async() 原生支持 asyncio，但仍需 patch_stdout 解决输出乱序问题。
session = PromptSession(">>> ")


async def input_reader(input_queue: asyncio.Queue[str]) -> None:
    """持续读取用户输入并推入队列，LLM 执行期间输入不会丢失。"""
    while True:
        user_input = await session.prompt_async()
        if user_input == _EXIT:
            # 取消所有任务，退出事件循环
            for task in asyncio.all_tasks():
                task.cancel()
            return
        await input_queue.put(user_input)
        if input_queue.qsize() > 1:
            items = list(input_queue._queue)  # asyncio.Queue 内部用 deque
            previews = [str(item)[:10] for item in items[1:]]
            print(f"[等待队列 {len(previews)} 条: {' | '.join(previews)}]", flush=True)


async def main() -> None:
    # asyncio.Queue 在单线程事件循环内天然安全，无需额外锁。
    input_queue: asyncio.Queue[str] = asyncio.Queue()

    asyncio.create_task(input_reader(input_queue))

    while True:
        # 挂起等待至少一条输入
        user_input = await input_queue.get()
        messages.append(HumanMessage(content=user_input))

        # 再把队列里积压的其余消息一并取出（非阻塞）
        # asyncio 单线程，empty() 与 get_nowait() 之间不存在竞态，绝对安全。
        while not input_queue.empty():
            messages.append(HumanMessage(content=input_queue.get_nowait()))

        # 所有积压消息合并后，只调用一次 LLM（全程异步，不阻塞事件循环）
        ai_message: AIMessage = await llm.ainvoke(messages)
        await asyncio.sleep(random.uniform(0, 5))  # 模拟 LLM 慢速输出

        messages.append(ai_message)
        print(f"\nAI: \n{ai_message.content}\n")


if __name__ == "__main__":
    try:
        with patch_stdout():
            asyncio.run(main())
    except asyncio.CancelledError:
        pass
