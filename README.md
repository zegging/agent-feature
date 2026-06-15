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

## Undo

在我们使用 Agent 的时候，总是需要撤回一些操作或者变更，就像 Ctrl + Z 或者是 git rollback 一样。但是 Agent 的实现也颇为不易。

[undo.py](/src/undo.py) 实现了一套确定性的 undo 系统。核心原则是：**undo 是 runtime 能力，不是 prompt 能力**。由 Agent Runtime 在每次工具调用前后记录副作用，再由 `UndoManager` 确定性地回滚。

```txt
User Prompt
  │
  ├─► 创建 Checkpoint（记录 message_cursor + effect_cursor）
  │
  └─► LLM 产生 tool_call
        │
        ├─► 执行前：_hash_file() 读取 before_hash，_put_blob() 存入内容寻址 blob store
        ├─► 执行：写入文件
        └─► 执行后：_put_blob() 存入 after_hash，追加 ToolCallRecord → _journal

/undo
  │
  └─► UndoManager.undo_turn()
        ├─► 从 Checkpoint 取 effect_cursor，找到本 turn 所有 ToolCallRecord
        ├─► 逆序遍历每条 FileEffect
        │     ├─► 冲突检测：current_hash == expected_current_hash？
        │     │     └─► 不等 → ConflictError（用户在 Agent 写完后又手动改了文件）
        │     └─► 恢复：before_hash == None → 删除文件，否则 _get_blob() 写回原内容
        └─► 截断 session 到 message_cursor（可选）
```

### 数据模型

| 结构 | 职责 |
|---|---|
| `Checkpoint` | 每次 user prompt 前的快照点，记录 `message_cursor`（session 长度）和 `effect_cursor`（journal 长度） |
| `ToolCallRecord` | 一次工具调用的完整记录：`turn_id`、`status`、`reversibility`、`file_effects[]` |
| `FileEffect` | 单个文件的变化：`before_hash`、`after_hash`、`expected_current_hash`（用于冲突检测） |
| Blob Store | 内容寻址的内存存储，SHA-256 为 key，before/after 内容各存一份，重复内容不浪费 |

### 三种 undo 粒度

| 命令 | 模式 | 行为 |
|---|---|---|
| `/undo` | `code_and_conversation` | 恢复文件 + 截断对话（最常用） |
| `/undo --code-only` | `code_only` | 只恢复文件，保留对话 |
| `/undo --conversation-only` | `conversation_only` | 只截断对话，不动文件 |
| `/redo` | — | 重新 apply 存储的 after-state blob，不重新调用 LLM |
| `/effects` | — | 查看当前 turn 所有工具调用的副作用日志 |

### 冲突检测

undo 时会比较 `current_hash == expected_current_hash`（即 Agent 写完时的 hash）。如果用户在 Agent 修改之后又手动编辑了文件，会抛出 `ConflictError` 而非静默覆盖：

```txt
  !!  CONFLICT — 'src/foo.py': file was modified after agent edit
      expected hash: 88446f07...
      actual hash:   e3376da7...
```