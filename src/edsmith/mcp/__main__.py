"""Top-level edSmith MCP server — aggregates all domain tools.

Run with:
    python -m edsmith.mcp
or via the CLI:
    edsmith start-server
"""

import logging
import os
import sys

# Allow local dev to override the installed package without reinstalling the plugin.
# Set EDSMITH_DEV_PATH to the repo's src/ directory in .claude/settings.local.json.
_dev_path = os.environ.get("EDSMITH_DEV_PATH")
if _dev_path:
    sys.path.insert(0, _dev_path)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from fastmcp import FastMCP

from edsmith.chief_examiner.mcp import (
    register_approve_proposal,
    register_chief_examiner,
    register_reject_proposal,
)
from edsmith.session.mcp import register_init_session
from edsmith.tools.mcp import (
    register_aoa_stats,
    register_complexity_stats,
    register_discourse_analysis,
    register_grammar_check,
)

mcp = FastMCP("edsmith")

_ = register_init_session(mcp)
_ = register_grammar_check(mcp)
_ = register_aoa_stats(mcp)
_ = register_complexity_stats(mcp)
_ = register_discourse_analysis(mcp)
_ = register_chief_examiner(mcp)
_ = register_approve_proposal(mcp)
_ = register_reject_proposal(mcp)


def main():
    try:
        mcp.run(transport="stdio")
    except (BrokenPipeError, EOFError):
        sys.exit(0)
    except Exception as e:
        print(f"edsmith MCP server error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
