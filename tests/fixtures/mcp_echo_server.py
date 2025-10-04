import sys, os
try:
    # Expected FastMCP 2.0 API (Python)
    from fastmcp import FastMCP
except Exception as e:
    # If API not as expected, fail loudly so the test can catch environment issues.
    raise

# Create a basic server instance
mcp = FastMCP(name="echo-server")

@mcp.tool(name="mcp_echo")
def mcp_echo(text: str) -> str:
    """Echoes the input text."""
    return text

def main(pidfile_path: str):
    # Write PID file to help the test validate lifecycle
    with open(pidfile_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
        f.flush()
    try:
        # Run stdio server; when the client disconnects, this should return.
        mcp.run()
    finally:
        try:
            os.remove(pidfile_path)
        except Exception:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: echo_server.py <pidfile>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
