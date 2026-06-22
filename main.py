"""
Main entry point for the FastAPI application
Run with: uvicorn main:app --reload
"""
import uvicorn
import sys
import asyncio
import os

# Fix for Windows: Playwright needs SelectorEventLoop instead of ProactorEventLoop
# MUST be set before uvicorn creates any event loop
if sys.platform == "win32":
    # Set the policy before any event loop is created
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Also set it as the default for new loops
    import os
    os.environ['PYTHONASYNCIODEBUG'] = '1'

if __name__ == "__main__":
    # On Windows, disable reload to avoid subprocess event loop issues
    # The scraper uses thread-based approach to handle Playwright
    use_reload = sys.platform != "win32"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=use_reload
    )

