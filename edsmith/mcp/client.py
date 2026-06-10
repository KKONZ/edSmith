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
import base64
import io
import json
from typing import Callable

import pandas as pd
from mcp import ClientSession, types
from mcp.client.sse import sse_client


def _df_to_b64(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return base64.b64encode(buf.getvalue()).decode()


class MCPClient:
    """Thin wrapper around the edsmith-training MCP server.

    DataFrames are serialised to base64-encoded parquet and sent inline so
    the server never needs to access the local filesystem.
    """

    def __init__(self, server_url: str) -> None:
        self._url = server_url

    # ------------------------------------------------------------------
    # Public callables
    # ------------------------------------------------------------------

    @property
    def trainer_fn(self) -> Callable:
        """Synchronous callable: (feedback_df, scorer_config, output_dir) -> model_path."""
        def _train(
            feedback_df: pd.DataFrame,
            scorer_config,
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
        result_json = await self._call_tool(
            "train_scorer",
            {
                "feedback_data": _df_to_b64(feedback_df),
                "scorer_config": json.dumps(scorer_config.model_dump()),
                "output_dir": output_dir,
            },
        )
        return json.loads(result_json)["model_path"]

    async def _aevaluate(
        self,
        model_path: str,
        df: pd.DataFrame,
    ) -> tuple[list[float], list[float]]:
        result_json = await self._call_tool(
            "evaluate_scorer",
            {
                "model_path": model_path,
                "eval_data": _df_to_b64(df),
            },
        )
        result = json.loads(result_json)
        return result["y_true"], result["y_pred"]

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
