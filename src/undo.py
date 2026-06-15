import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from _preload import api_key, base_url

# ─── Config ───────────────────────────────────────────────────────────────────

WORKDIR = Path.cwd()
_EXIT = "/exit"
_UNDO = "/undo"
_UNDO_CODE = "/undo --code-only"
_UNDO_CONV = "/undo --conversation-only"
_REDO = "/redo"
_EFFECTS = "/effects"
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

print("Welcome to the chat interface:", flush=True)
print(f"  {_EXIT}            exit", flush=True)
print(f"  {_UNDO}            undo last turn (code + conversation)", flush=True)
print(f"  {_UNDO_CODE}  undo file changes only", flush=True)
print(f"  {_UNDO_CONV}  undo conversation only", flush=True)
print(f"  {_REDO}            redo last undone turn", flush=True)
print(f"  {_EFFECTS}         show tool-call effects for current turn", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Path safety
# ══════════════════════════════════════════════════════════════════════════════


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Blob store  (content-addressed, in-memory)
# ══════════════════════════════════════════════════════════════════════════════

_blobs: dict[str, bytes] = {}


def _put_blob(content: str) -> str:
    """Store content, return its SHA-256 hex digest."""
    data = content.encode()
    h = hashlib.sha256(data).hexdigest()
    _blobs[h] = data
    return h


def _get_blob(h: str) -> str:
    return _blobs[h].decode()


def _hash_file(path: Path) -> str | None:
    """SHA-256 of current file contents (UTF-8), or None if the file does not exist."""
    if not path.exists():
        return None
    data = path.read_text(encoding="utf-8").encode()
    return hashlib.sha256(data).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  Data model
# ══════════════════════════════════════════════════════════════════════════════

Reversibility = Literal["read_only", "file_snapshot", "irreversible"]
ToolStatus = Literal["succeeded", "failed", "undone", "redone"]
RewindMode = Literal["code_only", "conversation_only", "code_and_conversation"]


@dataclass
class FileEffect:
    path: str
    operation: Literal["create", "modify", "delete"]
    before_hash: str | None  # content hash before tool ran (None = didn't exist)
    after_hash: str | None  # content hash after tool ran  (None = deleted)
    expected_current_hash: (
        str | None
    )  # = after_hash at write time; used for conflict detection


@dataclass
class ToolCallRecord:
    id: str
    turn_id: str
    tool_name: str
    status: ToolStatus
    reversibility: Reversibility
    file_effects: list[FileEffect] = field(default_factory=list)


@dataclass
class Checkpoint:
    id: str
    turn_id: str
    message_cursor: int  # len(session) before this turn's messages were appended
    effect_cursor: int  # len(_journal) before this turn's records were appended


# ══════════════════════════════════════════════════════════════════════════════
#  Global state
# ══════════════════════════════════════════════════════════════════════════════

_journal: list[ToolCallRecord] = []
_checkpoints: list[Checkpoint] = []
_current_turn_id: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  Conflict error
# ══════════════════════════════════════════════════════════════════════════════


class ConflictError(Exception):
    def __init__(self, path: str, expected: str | None, actual: str | None) -> None:
        super().__init__(
            f"'{path}': file was modified after agent edit\n"
            f"  expected hash: {expected}\n"
            f"  actual hash:   {actual}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  UndoManager
# ══════════════════════════════════════════════════════════════════════════════


class UndoManager:
    """
    Manages undo / redo at turn granularity.

    Design:
      - Every mutating tool call appends a ToolCallRecord to _journal.
      - Every user prompt creates a Checkpoint (message_cursor, effect_cursor).
      - undo_turn() restores files in reverse order and optionally truncates the session.
      - redo() re-applies the stored after-state blobs.
      - Conflict detection: if the file was edited after the agent wrote it,
        raise ConflictError rather than silently overwriting.
    """

    def __init__(self) -> None:
        self._redo_stack: list[tuple[Checkpoint, list[ToolCallRecord]]] = []

    # ── internal helpers ────────────────────────────────────────────────────

    def _restore_file(
        self,
        effect: FileEffect,
        restore_hash: str | None,
        expected_hash: str | None,
    ) -> str:
        file_path = WORKDIR / effect.path
        current_hash = _hash_file(file_path)

        if current_hash != expected_hash:
            raise ConflictError(effect.path, expected_hash, current_hash)

        if restore_hash is None:
            file_path.unlink(missing_ok=True)
            return f"deleted '{effect.path}'"
        else:
            content = _get_blob(restore_hash)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"restored '{effect.path}'"

    def _apply_record(
        self,
        record: ToolCallRecord,
        direction: Literal["undo", "redo"],
    ) -> list[str]:
        lines: list[str] = []

        if record.reversibility == "read_only":
            return lines

        if record.reversibility == "file_snapshot":
            effects = (
                list(reversed(record.file_effects))
                if direction == "undo"
                else record.file_effects
            )
            for effect in effects:
                restore_hash = (
                    effect.before_hash if direction == "undo" else effect.after_hash
                )
                expected_hash = (
                    effect.expected_current_hash
                    if direction == "undo"
                    else effect.before_hash
                )
                try:
                    msg = self._restore_file(effect, restore_hash, expected_hash)
                    lines.append(f"  ok  {msg}")
                except ConflictError as e:
                    lines.append(f"  !!  CONFLICT — {e}")

        elif record.reversibility == "irreversible":
            lines.append(f"  --  irreversible, skipped: {record.tool_name}")

        return lines

    # ── public API ──────────────────────────────────────────────────────────

    def undo_turn(
        self,
        session: list[BaseMessage],
        mode: RewindMode = "code_and_conversation",
    ) -> list[str]:
        """Undo the most recent turn. Returns status lines."""
        if not _checkpoints:
            return ["Nothing to undo."]

        cp = _checkpoints[-1]
        turn_records = [
            r for r in _journal[cp.effect_cursor :] if r.status == "succeeded"
        ]
        lines = [f"Undoing turn {cp.turn_id[:8]} ({mode})..."]

        if mode in ("code_only", "code_and_conversation"):
            for record in reversed(turn_records):
                lines.extend(self._apply_record(record, "undo"))
                record.status = "undone"

        if mode in ("conversation_only", "code_and_conversation"):
            del session[cp.message_cursor :]
            lines.append(
                f"  ok  conversation rolled back to message {cp.message_cursor}"
            )

        self._redo_stack.append((cp, turn_records))
        _checkpoints.pop()
        return lines

    def redo(self, session: list[BaseMessage]) -> list[str]:
        """Re-apply the last undone turn."""
        if not self._redo_stack:
            return ["Nothing to redo."]

        cp, records = self._redo_stack.pop()
        lines = [f"Redoing turn {cp.turn_id[:8]}..."]

        for record in records:
            lines.extend(self._apply_record(record, "redo"))
            record.status = "redone"

        _checkpoints.append(cp)
        return lines

    def show_effects(self) -> list[str]:
        """Show tool-call effects for the current (last) turn."""
        if not _checkpoints:
            return ["No turns recorded yet."]

        cp = _checkpoints[-1]
        records = _journal[cp.effect_cursor :]
        if not records:
            return ["No tool calls in current turn."]

        lines = [f"Effects for turn {cp.turn_id[:8]}:"]
        for r in records:
            lines.append(f"  [{r.reversibility}] {r.tool_name}  status={r.status}")
            for e in r.file_effects:
                lines.append(f"      {e.operation:6}  {e.path}")
        return lines


undo_manager = UndoManager()

# ══════════════════════════════════════════════════════════════════════════════
#  Journal helpers called by tool implementations
# ══════════════════════════════════════════════════════════════════════════════


def _begin_record(tool_name: str, reversibility: Reversibility) -> ToolCallRecord:
    record = ToolCallRecord(
        id=str(uuid.uuid4()),
        turn_id=_current_turn_id,
        tool_name=tool_name,
        status="succeeded",  # will be overwritten to "failed" in except
        reversibility=reversibility,
    )
    _journal.append(record)
    return record


def _snapshot_and_run_write(
    record: ToolCallRecord, file_path: Path, rel_path: str, content: str
) -> None:
    """Capture before-state, write the file, record FileEffect. Must be called inside try/finally."""
    before_hash = _hash_file(file_path)
    operation: Literal["create", "modify", "delete"] = (
        "create" if before_hash is None else "modify"
    )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    after_hash = _put_blob(content)
    if before_hash is not None:
        # ensure before-blob is also stored
        _put_blob(
            _get_blob(before_hash)
            if before_hash in _blobs
            else file_path.read_text(encoding="utf-8")
        )

    record.file_effects.append(
        FileEffect(
            path=rel_path,
            operation=operation,
            before_hash=before_hash,
            after_hash=after_hash,
            expected_current_hash=after_hash,
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Tool Protocol
# ══════════════════════════════════════════════════════════════════════════════


class ToolDefinition(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def run(self, *args, **kwargs) -> str: ...


# ══════════════════════════════════════════════════════════════════════════════
#  Tools
# ══════════════════════════════════════════════════════════════════════════════


class Write:
    name = "write_file"
    description = "Write content to a file."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        record = _begin_record(self.name, "file_snapshot")
        try:
            file_path = safe_path(path)
            rel = str(Path(path))

            # snapshot before-state so undo can restore it
            before_hash = _hash_file(file_path)
            if before_hash is not None:
                _put_blob(
                    file_path.read_text(encoding="utf-8")
                )  # ensure blob is stored
            operation: Literal["create", "modify", "delete"] = (
                "create" if before_hash is None else "modify"
            )

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            after_hash = _put_blob(content)

            record.file_effects.append(
                FileEffect(
                    path=rel,
                    operation=operation,
                    before_hash=before_hash,
                    after_hash=after_hash,
                    expected_current_hash=after_hash,
                )
            )
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            record.status = "failed"
            return f"Error: {e}"


class Read:
    name = "read_file"
    description = "Read file contents."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
    }

    def run(self, path: str, limit: int | None = None) -> str:
        _begin_record(self.name, "read_only")
        try:
            lines = safe_path(path).read_text(encoding="utf-8").splitlines()
            if limit and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
            return "\n".join(lines)
        except Exception as e:
            _journal[-1].status = "failed"
            return f"Error: {e}"


class Edit:
    name = "edit_file"
    description = "Replace exact text in a file once."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def run(self, path: str, old_text: str, new_text: str) -> str:
        record = _begin_record(self.name, "file_snapshot")
        try:
            file_path = safe_path(path)
            rel = str(Path(path))
            text = file_path.read_text(encoding="utf-8")

            if old_text not in text:
                record.status = "failed"
                return f"Error: text not found in {path}"

            before_hash = _put_blob(text)  # store + get hash in one step
            new_text_full = text.replace(old_text, new_text, 1)
            file_path.write_text(new_text_full, encoding="utf-8")
            after_hash = _put_blob(new_text_full)

            record.file_effects.append(
                FileEffect(
                    path=rel,
                    operation="modify",
                    before_hash=before_hash,
                    after_hash=after_hash,
                    expected_current_hash=after_hash,
                )
            )
            return f"Edited {path}"
        except Exception as e:
            record.status = "failed"
            return f"Error: {e}"


class Glob:
    name = "glob"
    description = "List files matching a glob pattern."
    input_schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    }

    def run(self, pattern: str) -> str:
        _begin_record(self.name, "read_only")
        import glob as g

        try:
            results = [
                m
                for m in g.glob(pattern, root_dir=WORKDIR)
                if (WORKDIR / m).resolve().is_relative_to(WORKDIR)
            ]
            return "\n".join(results) if results else "(no matches)"
        except Exception as e:
            _journal[-1].status = "failed"
            return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  LLM + tool registry
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatOpenAI(model="gpt-5.2", api_key=api_key, base_url=base_url)

TOOLS: list[ToolDefinition] = [Write(), Read(), Edit(), Glob()]
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        },
    }
    for t in TOOLS
]
TOOL_HANDLERS = {t.name: t.run for t in TOOLS}
llm_with_tools = llm.bind_tools(OPENAI_TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    global _current_turn_id

    session: list[BaseMessage] = [SystemMessage(content=SYSTEM)]

    while True:
        user_input = input(">>> ").strip()

        # ── meta commands ────────────────────────────────────────────────────
        if user_input == _EXIT:
            print("Exiting.", flush=True)
            break

        if user_input in (_UNDO, _UNDO_CODE, _UNDO_CONV):
            mode: RewindMode = (
                "code_only"
                if user_input == _UNDO_CODE
                else "conversation_only"
                if user_input == _UNDO_CONV
                else "code_and_conversation"
            )
            for line in undo_manager.undo_turn(session, mode):
                print(line, flush=True)
            continue

        if user_input == _REDO:
            for line in undo_manager.redo(session):
                print(line, flush=True)
            continue

        if user_input == _EFFECTS:
            for line in undo_manager.show_effects():
                print(line, flush=True)
            continue

        if not user_input:
            continue

        # ── new turn ─────────────────────────────────────────────────────────
        _current_turn_id = str(uuid.uuid4())
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            turn_id=_current_turn_id,
            message_cursor=len(session),  # save position before appending
            effect_cursor=len(_journal),
        )
        _checkpoints.append(checkpoint)

        session.append(HumanMessage(content=user_input))

        # tool-call loop: keep invoking until LLM stops requesting tools
        while True:
            response = await llm_with_tools.ainvoke(session)
            session.append(response)

            if not response.tool_calls:
                print(f"AI: {response.content}", flush=True)
                break

            for tc in response.tool_calls:
                handler = TOOL_HANDLERS.get(tc["name"])
                result = (
                    handler(**tc["args"]) if handler else f"Unknown tool: {tc['name']}"
                )
                print(f"  [tool:{tc['name']}]\n  {result}\n", flush=True)
                session.append(ToolMessage(content=result, tool_call_id=tc["id"]))


if __name__ == "__main__":
    asyncio.run(main())
