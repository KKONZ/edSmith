"""Top-level edSmith MCP server — aggregates all domain tools.

Run with:
    python -m edsmith.mcp
or via the CLI:
    edsmith start-server
"""

from fastmcp import FastMCP

mcp = FastMCP("edsmith")

# Domain tool modules are imported and registered here as they are added:
# from edsmith.tools.mcp.tools import *       # linguistic feature tools
# from edsmith.examiner.mcp.tools import *    # run_examiner_pass
# from edsmith.chief_examiner.mcp.tools import *  # run_chief_examiner, approve/reject

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
