"""Check X login and scripting permissions in the user's Arc browser."""

from research_digest.retrieval.x_source import login
import sys


if __name__ == "__main__":
    try:
        login()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
