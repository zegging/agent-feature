# Interesting Agent feature

a project for interesting agent feature, implement or creative

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