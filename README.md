# Interesting Agent feature

A collection of interesting agent features — practical implementations and creative experiments.

## Run

`uv sync` 即可。

## Message Queue

在我们使用 Agent 的时候，有一个非常好用的功能：当 tool 在执行调用或者 LLM 在生成文本的时候我们可以继续输入自己想说的话，不管是输入一些提示信息又或者是输入一些纠偏（不想停止正在进行的工具调用）。

[message queue](/src/message_queue.py) 就是这个功能的简单实现，引入 `prompt-toolkit` 稍微处理了一下控制台绘制的问题。

```txt
┌──────────────────┐        put        ┌════════════════════════┐        get        ┌──────────────────┐
│ input_reader 线程│ ───────────────►  │      input_queue       │ ───────────────►  │      主线程       │
│ 持续接收输入      │                   │  msg1 │ msg2 │ msg3    │                   │ 批量取出并调用 LLM │
└──────────────────┘                   │                        │                   └──────────────────┘
                                       │  缓存输入               │
                                       │  解耦输入和 LLM 执行     │
                                       │  防止慢响应期间丢消息    │
                                       └════════════════════════┘
```

核心关系：`input_thread` 只负责 **写** 队列，主线程只负责 **读** 队列，两者通过 `input_queue` 解耦，互不阻塞。

### Async 版本

[message_queue_async.py](/src/message_queue_async.py) 是全程异步的等价实现，用 `asyncio` 替代多线程：

| | threading 版 | asyncio 版 |
|---|---|---|
| 并发模型 | 多线程 | 单线程事件循环 |
| 输入读取 | `session.prompt()` + `Thread` | `await session.prompt_async()` + `create_task` |
| 队列 | `queue.Queue` | `asyncio.Queue` |
| LLM 调用 | `llm.invoke()`（阻塞） | `await llm.ainvoke()`（非阻塞） |
| 竞态风险 | 理论上存在（实际安全） | 完全不存在 |

两个版本都依赖 `patch_stdout` 解决控制台输出乱序问题，这与并发模型无关——`print()` 输出时需要清除输入行、打印后重绘提示符，缺少它就会出现输出与输入行交错的现象。