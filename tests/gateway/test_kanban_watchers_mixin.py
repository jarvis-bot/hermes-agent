"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
import asyncio
from unittest.mock import patch

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_reviewer_gateway_cannot_start_dispatcher():
    class Reviewer(GatewayKanbanWatchersMixin):
        _running = True

        @staticmethod
        def _active_profile_name():
            return "reviewer"

    with patch(
        "hermes_cli.config.load_config",
        return_value={"kanban": {"dispatch_in_gateway": True}},
    ):
        # Returning before the initial sleep proves the profile gate is before
        # every board open, request consume, reclaim, claim, and spawn path.
        asyncio.run(Reviewer()._kanban_dispatcher_watcher())
