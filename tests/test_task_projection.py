import unittest

from simulator.task_projection import project_task_lifecycles


def event(event_type: str, time_s: int, task_id: int = 1, **details: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "type": event_type,
        "time_s": time_s,
        "task_id": task_id,
        **details,
    }


class TaskProjectionTests(unittest.TestCase):
    def test_merges_deferred_task_lifecycle_by_task_id(self) -> None:
        lifecycle = project_task_lifecycles(
            [
                event("task_generated", 0, source_sat=2),
                event("task_assigned", 0, source_sat=2, target_sat=2, mode="defer"),
                event("task_deferred", 0, source_sat=2),
                event("task_assigned", 30, source_sat=2, target_sat=7, mode="offload"),
                event("task_completed", 30, source_sat=2, target_sat=7, mode="offload"),
            ]
        )[0]

        self.assertEqual(lifecycle.task_id, 1)
        self.assertEqual(lifecycle.created_time_s, 0)
        self.assertEqual(lifecycle.status, "completed")
        self.assertEqual((lifecycle.source_sat, lifecycle.target_sat), (2, 7))
        self.assertEqual(lifecycle.mode, "offload")
        self.assertEqual(len(lifecycle.events), 5)
        self.assertTrue(lifecycle.completed)
        self.assertEqual(lifecycle.terminal_event["type"], "task_completed")

    def test_preserves_generated_order_and_pending_status(self) -> None:
        lifecycles = project_task_lifecycles(
            [
                event("task_generated", 0, task_id=8, source_sat=None),
                event("task_waiting_for_coverage", 30, task_id=8),
                event("task_generated", 30, task_id=3, source_sat=1),
            ]
        )

        self.assertEqual([lifecycle.task_id for lifecycle in lifecycles], [8, 3])
        self.assertEqual(lifecycles[0].status, "pending")
        self.assertIsNone(lifecycles[0].source_sat)
        self.assertIsNone(lifecycles[0].terminal_event)

    def test_rejects_event_before_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "must start with task_generated"):
            project_task_lifecycles([event("task_completed", 30)])

    def test_rejects_event_after_terminal_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "after its terminal event"):
            project_task_lifecycles(
                [
                    event("task_generated", 0),
                    event("task_failed", 30),
                    event("task_assigned", 60),
                ]
            )


if __name__ == "__main__":
    unittest.main()
