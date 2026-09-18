"""
crawler.py
-----------
Captures screenshots + DOM snapshots for a list of website flows,
so the Persona Engine can later run AI checks against them.

Token-shortage mitigation baked in (per the mitigation notes):
  - Caches captures by URL+flow so re-running the crawler while
    debugging doesn't re-hit the site or waste time.
  - "Batch, don't loop wastefully": each flow's screens are captured
    together and saved as ONE JSON bundle per flow, so the persona
    engine can send one full flow's context in a single LLM call
    instead of many small back-and-forth calls.
  - MOCK_MODE: skips real browser capture entirely and loads a fake
    bundle, so frontend/UI work never touches the crawler or burns
    time/tokens re-crawling during iteration.

Requirements:
    pip install playwright
    playwright install chromium
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from playwright.async_api import async_playwright, Page

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

OUTPUT_DIR = Path("captures")
CACHE_FILE = Path("captures/_cache_index.json")
MOCK_MODE = os.getenv("CRAWLER_MOCK_MODE", "false").lower() == "true"
VIEWPORT = {"width": 1440, "height": 900}


@dataclass
class FlowStep:
    """One screen inside a flow (e.g. step 2 of the checkout flow)."""
    flow_id: str
    step_index: int
    url: str
    screenshot_path: str
    dom_snapshot: str
    captured_at: str


@dataclass
class Flow:
    """A named user journey made up of ordered URLs to visit."""
    flow_id: str
    urls: List[str]


# Hardcode your flows here for the hackathon demo rather than building
# a full auto-discovery crawler — far more reliable under time pressure.
FLOWS: List[Flow] = [
    Flow(
        flow_id="checkout_flow",
        urls=[
            "https://example.com/",
            "https://example.com/product/1",
            "https://example.com/cart",
            "https://example.com/checkout",
            "https://example.com/order-confirmation",
        ],
    ),
    Flow(
        flow_id="signup_flow",
        urls=[
            "https://example.com/signup",
            "https://example.com/verify-email",
            "https://example.com/onboarding",
        ],
    ),
]


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def _load_cache_index() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache_index(index: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(index, indent=2))


def _cache_key(flow_id: str, url: str) -> str:
    return hashlib.sha256(f"{flow_id}:{url}".encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# Mock mode (for frontend / persona-engine dev without a live crawl)
# ----------------------------------------------------------------------

def load_mock_bundle(flow_id: str) -> List[FlowStep]:
    """Returns a small fake FlowStep list so downstream code (persona
    engine, UI) can be developed and tested without ever launching a
    browser or spending time/tokens on a real crawl."""
    fake_steps = []
    for i in range(1, 4):
        fake_steps.append(
            FlowStep(
                flow_id=flow_id,
                step_index=i,
                url=f"https://example.com/mock-step-{i}",
                screenshot_path=f"captures/mock/{flow_id}_{i}.png",
                dom_snapshot="<html><body>Mock DOM content</body></html>",
                captured_at=datetime.now(timezone.utc).isoformat(),
            )
        )
    return fake_steps


# ----------------------------------------------------------------------
# Real capture logic
# ----------------------------------------------------------------------

async def _capture_single_page(page: Page, flow_id: str, step_index: int, url: str) -> FlowStep:
    """Navigates to a URL and captures a screenshot + the rendered DOM."""
    await page.goto(url, wait_until="networkidle", timeout=30_000)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_path = OUTPUT_DIR / f"{flow_id}_{step_index}.png"
    await page.screenshot(path=str(screenshot_path), full_page=True)

    dom_snapshot = await page.content()

    return FlowStep(
        flow_id=flow_id,
        step_index=step_index,
        url=url,
        screenshot_path=str(screenshot_path),
        dom_snapshot=dom_snapshot,
        captured_at=datetime.now(timezone.utc).isoformat(),
    )


async def capture_flow(flow: Flow, cache_index: dict, force: bool = False) -> List[FlowStep]:
    """Captures every step of a single flow, skipping steps already
    cached unless force=True. Steps are captured sequentially within
    a flow (order matters for a user journey)."""
    steps: List[FlowStep] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        for i, url in enumerate(flow.urls, start=1):
            key = _cache_key(flow.flow_id, url)

            if not force and key in cache_index:
                cached = cache_index[key]
                steps.append(FlowStep(**cached))
                print(f"[cache hit] {flow.flow_id} step {i}: {url}")
                continue

            print(f"[capturing] {flow.flow_id} step {i}: {url}")
            try:
                step = await _capture_single_page(page, flow.flow_id, i, url)
            except Exception as e:
                print(f"  ! failed to capture {url}: {e}")
                continue

            steps.append(step)
            cache_index[key] = asdict(step)

        await browser.close()

    return steps


def save_flow_bundle(flow_id: str, steps: List[FlowStep]) -> Path:
    """Saves all steps of a flow as ONE JSON file — this is the bundle
    the persona engine sends in a single batched call per flow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = OUTPUT_DIR / f"{flow_id}_bundle.json"
    bundle_path.write_text(json.dumps([asdict(s) for s in steps], indent=2))
    print(f"[saved] {bundle_path} ({len(steps)} steps)")
    return bundle_path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

async def run_crawler(flows: Optional[List[Flow]] = None, force: bool = False) -> None:
    flows = flows or FLOWS
    cache_index = _load_cache_index()

    for flow in flows:
        if MOCK_MODE:
            print(f"[MOCK MODE] Skipping real crawl for '{flow.flow_id}'")
            steps = load_mock_bundle(flow.flow_id)
        else:
            steps = await capture_flow(flow, cache_index, force=force)

        save_flow_bundle(flow.flow_id, steps)

    if not MOCK_MODE:
        _save_cache_index(cache_index)


if __name__ == "__main__":
    # Set CRAWLER_MOCK_MODE=true as an env var to test the pipeline
    # end-to-end without launching a real browser.
    asyncio.run(run_crawler())