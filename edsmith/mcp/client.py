"""MCP client wrapper — connects to the Colab training server.

Usage:
    from edsmith.mcp.client import MCPClient

    client = MCPClient("https://<tunnel-id>.trycloudflare.com/sse")
    orchestrator = Orchestrator(
        ...
        trainer_fn=client.trainer_fn,
        evaluator_fn=client.evaluator_fn,
    )

Or via CLI:
    edsmith run-session --config session.yaml --mcp-url https://<tunnel-id>.trycloudflare.com/sse
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import pandas as pd
from mcp import ClientSession, types
from mcp.client.sse import sse_client


class MCPClient:
    """Thin wrapper around the edsmith-training MCP server.

    Exposes ``trainer_fn`` and ``evaluator_fn`` as synchronous callables
    so they drop in directly as Orchestrator arguments.
    """

    def __init__(self, server_url: str) -> None:
        # e.g. "https://abc123.trycloudflare.com/sse"
        self._url = server_url

    # ------------------------------------------------------------------
    # Public callables
    # ------------------------------------------------------------------

    @property
    def trainer_fn(self) -> Callable:
        """Synchronous callable: (feedback_df, scorer_config, output_dir) -> model_path."""
        def _train(
            feedback_df: pd.DataFrame,
            scorer_config,  # ScorerConfig instance
            output_dir: str,
        ) -> str:
            return asyncio.run(self._atrain(feedback_df, scorer_config, output_dir))
        return _train

    @property
    def evaluator_fn(self) -> Callable:
        """Synchronous callable: (model_path, df) -> (y_true, y_pred)."""
        def _evaluate(
            model_path: str,
            df: pd.DataFrame,
        ) -> tuple[list[float], list[float]]:
            return asyncio.run(self._aevaluate(model_path, df))
        return _evaluate

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def _atrain(
        self,
        feedback_df: pd.DataFrame,
        scorer_config,
        output_dir: str,
    ) -> str:
        # Write feedback to a temp parquet on Drive so the server can read it
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            feedback_df.to_parquet(tmp_path, index=False)
            result_json = await self._call_tool(
                "train_scorer",
                {
                    "feedback_path": tmp_path,
                    "scorer_config": json.dumps(scorer_config.model_dump()),
                    "output_dir": output_dir,
                },
            )
            return json.loads(result_json)["model_path"]
        finally:
            os.unlink(tmp_path)

    async def _aevaluate(
        self,
        model_path: str,
        df: pd.DataFrame,
    ) -> tuple[list[float], list[float]]:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            df.to_parquet(tmp_path, index=False)
            result_json = await self._call_tool(
                "evaluate_scorer",
                {"model_path": model_path, "eval_data_path": tmp_path},
            )
            result = json.loads(result_json)
            return result["y_true"], result["y_pred"]
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Low-level MCP call
    # ------------------------------------------------------------------

    async def _call_tool(self, name: str, arguments: dict) -> str:
        async with sse_client(self._url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        return content.text
                raise RuntimeError(f"Tool {name!r} returned no text content")
