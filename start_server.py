"""
Startup script that sets Windows event loop policy BEFORE uvicorn starts
This must be run instead of main.py on Windows
"""
import sys
import asyncio
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
print(f"[Startup] Loaded environment variables from .env file")

APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "8001"))

# CRITICAL: Set event loop policy BEFORE any other imports
if sys.platform == "win32":
    # Set the policy as early as possible
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Patch asyncio.run and Runner to ensure SelectorEventLoop is used
    original_run = asyncio.run
    original_new_event_loop = asyncio.new_event_loop
    
    def patched_new_event_loop():
        """Force SelectorEventLoop creation"""
        policy = asyncio.get_event_loop_policy()
        if isinstance(policy, asyncio.WindowsProactorEventLoopPolicy):
            # Force SelectorEventLoop policy
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            policy = asyncio.get_event_loop_policy()
        return policy.new_event_loop()
    
    def patched_run(main, *, debug=None):
        """Ensure policy is set before asyncio.run creates event loop"""
        # Force policy before run
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # Patch new_event_loop temporarily
        asyncio.new_event_loop = patched_new_event_loop
        try:
            return original_run(main, debug=debug)
        finally:
            # Restore original
            asyncio.new_event_loop = original_new_event_loop
    
    asyncio.run = patched_run
    asyncio.new_event_loop = patched_new_event_loop
    
    # Set environment variable
    os.environ['PYTHONASYNCIODEBUG'] = '1'
    
    print("[Startup] WindowsSelectorEventLoopPolicy set and asyncio patched")

# Now import and run uvicorn
import uvicorn
from uvicorn import Config, Server

if __name__ == "__main__":
    # Verify policy is set
    if sys.platform == "win32":
        policy = asyncio.get_event_loop_policy()
        print(f"[Startup] Event loop policy: {type(policy).__name__}")
        if "Proactor" in type(policy).__name__:
            print("[Startup] WARNING: ProactorEventLoopPolicy detected! This will cause Playwright to fail.")
            # Force set SelectorEventLoop policy
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            print("[Startup] Forced WindowsSelectorEventLoopPolicy")
    
    # Use Server API directly to have more control over event loop
    config = Config(
        "api.app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False  # Disable reload to avoid subprocess event loop issues
    )
    
    server = Server(config)
    
    # Run server with explicit event loop control
    if sys.platform == "win32":
        # Create SelectorEventLoop explicitly and run server in it
        loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
        asyncio.set_event_loop(loop)
        print(f"[Startup] Created event loop: {type(loop).__name__}")
        
        try:
            loop.run_until_complete(server.serve())
        except KeyboardInterrupt:
            print("\n[Startup] Shutting down...")
        finally:
            loop.close()
    else:
        # On non-Windows, use standard uvicorn.run
        uvicorn.run(
            "api.app:app",
            host=APP_HOST,
            port=APP_PORT,
            reload=False
        )
