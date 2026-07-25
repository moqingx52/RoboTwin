#!/usr/bin/env python3
"""Tests for phase2b orchestrator skip propagation logic."""

import unittest


def should_skip_job(job, jobs, groups, promotions):
    """Mirror of Scheduler._should_skip_job for unit testing."""
    dependency = job.get("dependency")
    if dependency:
        if jobs[dependency]["status"] == "skipped":
            return True
    blocked_by = job.get("blocked_by")
    if blocked_by:
        promo = promotions.get(blocked_by, {})
        if promo.get("status") == "eliminated":
            return True
        blocked_group = groups.get(blocked_by)
        if blocked_group and blocked_group.get("status") == "skipped":
            return True
        if blocked_group:
            train_dep = blocked_group.get("dependency")
            if train_dep and jobs[train_dep]["status"] == "skipped":
                return True
    depends_on_group = job.get("depends_on_group")
    if depends_on_group:
        group = groups.get(depends_on_group)
        if group and group.get("status") == "skipped":
            return True
    return False


class SkipPropagationTest(unittest.TestCase):
    def test_should_skip_when_blocked_by_eliminated(self):
        jobs = {"train:next": {"id": "train:next", "status": "pending", "blocked_by": "screen:g:epoch5"}}
        groups = {}
        promotions = {"screen:g:epoch5": {"status": "eliminated"}}
        self.assertTrue(should_skip_job(jobs["train:next"], jobs, groups, promotions))

    def test_all_complete_counts_skipped(self):
        jobs = [{"status": "completed"}, {"status": "skipped"}]
        groups = [{"status": "skipped"}]
        terminal_job = all(j["status"] in ("completed", "skipped") for j in jobs)
        terminal_group = all(g["status"] in ("completed", "skipped") for g in groups)
        self.assertTrue(terminal_job and terminal_group)


if __name__ == "__main__":
    unittest.main()
