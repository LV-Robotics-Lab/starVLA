from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = Path(__file__).with_name("adapter.py")
SPEC = importlib.util.spec_from_file_location("prometheus_star_vla_adapter", ADAPTER_PATH)
assert SPEC and SPEC.loader
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class AdapterTest(unittest.TestCase):
    def test_doctor_is_structural_and_hardware_closed(self) -> None:
        report = ADAPTER.doctor()
        declared = ADAPTER.capabilities()
        self.assertTrue(report["ok"])
        self.assertFalse(report["imports_model_stack"])
        self.assertEqual(declared["capabilities"]["resume"], "weights_only")
        self.assertFalse(declared["capabilities"]["hardware_rollout_authorized"])

    def test_training_command_is_an_argv_array(self) -> None:
        command = ADAPTER.build_argv(
            "train",
            ["--config_yaml", "recipe.yaml", "--run_id", "smoke"],
            accelerate_args=["--num_processes", "2"],
        )
        self.assertEqual(
            command[:6],
            [
                "accelerate",
                "launch",
                "--num_processes",
                "2",
                "starVLA/training/train_starvla.py",
                "--config_yaml",
            ],
        )
        self.assertNotIn("bash", command)

    def test_resume_is_explicitly_weights_only(self) -> None:
        command = ADAPTER.build_argv(
            "resume",
            ["--config_yaml", "recipe.yaml"],
            checkpoint="/managed/checkpoints/steps_100_model.safetensors",
        )
        self.assertEqual(command[-4:], [
            "--trainer.pretrained_checkpoint",
            "/managed/checkpoints/steps_100_model.safetensors",
            "--trainer.is_resume",
            "false",
        ])

    def test_cli_plan_returns_json_without_launching_models(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                str(ADAPTER_PATH),
                "train",
                "--plan",
                "--",
                "--config_yaml",
                "recipe.yaml",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(process.stdout)
        self.assertFalse(payload["shell"])
        self.assertEqual(payload["argv"][:2], ["accelerate", "launch"])

    def test_no_universal_prepare_or_eval_is_claimed(self) -> None:
        stages = ADAPTER.capabilities()["stages"]
        self.assertFalse(stages["prepare"])
        self.assertFalse(stages["eval"])
        self.assertFalse(stages["export"])


if __name__ == "__main__":
    unittest.main()
