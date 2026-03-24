"""AWS Lambda handler — Mangum adapter for the Starlette ASGI app."""

from mangum import Mangum

from agent_memory.api import app

handler = Mangum(app, lifespan="off")
