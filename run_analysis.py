"""Command-line entrypoint for running the opportunity analysis workflow."""

import asyncio

from app.run_workflow import run_analysis_workflow


if __name__ == "__main__":
    asyncio.run(run_analysis_workflow())
