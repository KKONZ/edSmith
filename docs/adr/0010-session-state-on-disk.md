# Session State Persisted to Disk

All session state is persisted to disk under `{drive_path}/sessions/{session_id}/`. `SessionState` is a Pydantic model serialised as `state.json`. `HumanReviewProposal` records are serialised as `proposals/iter{n}.json`. Feedback parquets are written as `feedback_iter{n}.parquet`. Each MCP tool reads what it needs from disk and writes its outputs before returning.

**Why:** Each MCP tool must be independently re-entrant. Conversations can be interrupted, the human can close the terminal, Colab can time out, or a tool call can fail. Persisting to disk after every tool call means resumption is always possible by re-reading `state.json` and checking which files already exist for the current iteration. It also makes the session state inspectable by hand without running any code.

**Proposal lifecycle:** A `HumanReviewProposal` starts with `status = "pending"` when `run_chief_examiner` writes it. The human approves or rejects it in the Claude Code conversation, which calls `approve_proposal` or `reject_proposal`. These tools update the `status` field and apply or discard the proposed policies and strategy. The proposal file is a permanent record of every human decision in the session tree.

**Session identity and tree structure:** `session_id` is a UUID generated at session start. `parent_session_id` links a new session to the iteration it branched from, encoding the tree structure used by the MCTS reflection mode.

**Considered alternatives:** Storing state in a SQLite database was considered. Rejected because JSON files are simpler to inspect by hand, easier to version-control for debugging, and sufficient for the single-writer access pattern of a sequential session loop.
