import logging
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train
from train import load_checkpoint, save_checkpoint
from utils.parsers import build_parser
from utils.runtime import atomic_torch_save, seed_everything
from utils.user_identity import build_identity, validate_identity


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "last.pt"
        self.args = build_parser().parse_args([])

    def model(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.3), nn.Linear(4, 1))
        model.user_identity = build_identity(['author-a'])
        model.config = {"data_protocol": {"fingerprint": "test"},
                        "user_identity": model.user_identity}
        model.initialization_policy = 'train-mean-output-bias-v1'
        model.fresh_rng_policy = train.FRESH_RNG_POLICY
        return model, torch.optim.Adam(model.parameters(), lr=0.01)

    def epoch(self, model, optimizer):
        model.train()
        x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
        loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=True)
        for (batch,) in loader:
            optimizer.zero_grad()
            noise = random.random() + np.random.random() + torch.rand(())
            loss = (model(batch) - noise).square().mean()
            loss.backward()
            optimizer.step()

    def test_interrupted_training_matches_continuous_exactly(self):
        seed_everything(12)
        model, optimizer = self.model()
        self.epoch(model, optimizer)
        save_checkpoint(
            self.path, model, optimizer, 0, 0.5, self.args, model.config, {}, bad_epochs=3
        )
        self.epoch(model, optimizer)
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        next_random = (random.random(), np.random.random(), torch.rand(()))
        seed_everything(999)
        resumed, resumed_optimizer = self.model()
        state = load_checkpoint(
            str(self.path), resumed, resumed_optimizer, torch.device("cpu"), self.args
        )
        self.assertEqual(state, (1, 0.5, 3))
        self.epoch(resumed, resumed_optimizer)
        for key, value in resumed.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertEqual((random.random(), np.random.random()), next_random[:2])
        torch.testing.assert_close(torch.rand(()), next_random[2], rtol=0, atol=0)

    def test_resume_rejects_changed_batch_size(self):
        model, optimizer = self.model()
        save_checkpoint(self.path, model, optimizer, 0, 0.5, self.args, model.config, {})
        self.args.batch_size += 1
        with self.assertRaisesRegex(ValueError, "batch_size"):
            load_checkpoint(str(self.path), model, optimizer, torch.device("cpu"), self.args)

    def test_best_checkpoint_is_lightweight_and_not_resumable(self):
        model, optimizer = self.model()
        save_checkpoint(
            self.path, model, optimizer, 0, 0.5, self.args, model.config, {}, resumable=False
        )
        saved = torch.load(self.path, weights_only=False)
        self.assertNotIn("optimizer_state_dict", saved)
        with self.assertRaisesRegex(ValueError, "complete resume state"):
            load_checkpoint(str(self.path), model, optimizer, torch.device("cpu"), self.args)

    def test_failed_save_preserves_previous_checkpoint(self):
        atomic_torch_save({"value": 1}, self.path)
        with patch("utils.runtime.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                atomic_torch_save({"value": 2}, self.path)
        self.assertEqual(torch.load(self.path)["value"], 1)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_main_retention_and_resumed_early_stop(self):
        class Split(list):
            protocol = {"fingerprint": "test"}
            original_user_ids = ['author-a']

        class Model(nn.Linear):
            def __init__(self, args, config, meta_dim, dropout):
                super().__init__(2, 1)
                self.config = config
                self.user_identity = validate_identity(config['user_identity'])

            @property
            def regressor(self):
                return (self,)

        split = Split([Namespace(meta_features=torch.empty(0), y=torch.tensor([2.]))])
        metrics = [{"mse": 0.5, "mae": 0.4, "src": 0.8}, {"mse": 0.6, "mae": 0.5, "src": 0.7}]
        cli = [
            "train.py",
            "--output-dir",
            self.temp.name,
            "--run-name",
            "check",
            "--device",
            "cpu",
            "--patience",
            "1",
            "--epochs",
            "5",
        ]
        try:
            with (
                patch.object(train, "load_config", return_value={}),
                patch.object(train, "meta_fields_for", return_value=[]),
                patch.object(train, "read_data", return_value=(split, split)),
                patch.object(train, "move_items_to_device", return_value=()),
                patch.object(train, "JRPP", Model),
                patch.object(train, "initialize_user_identity", return_value=None),
                patch.object(
                    train, "train_one_epoch", return_value={"loss": 1.0, "mse_loss": 1.0}
                ) as epoch,
                patch.object(train, "evaluate", side_effect=metrics),
                patch.object(sys, "argv", cli),
            ):
                train.main()
                self.assertEqual(epoch.call_count, 2)
                root = Path(self.temp.name) / "icip" / "check"
                last = root / "checkpoints" / "JRPP_last.pt"
                self.assertEqual(
                    sorted(p.name for p in root.rglob("*.pt")), ["JRPP_best.pt", "JRPP_last.pt"]
                )
                saved = torch.load(last, weights_only=False)
                self.assertEqual(saved["bad_epochs"], 1)
                epoch.reset_mock()
                resume_cli = [
                    "train.py",
                    "--output-dir",
                    self.temp.name,
                    "--device",
                    "cpu",
                    "--patience",
                    "1",
                    "--epochs",
                    "10",
                    "--resume-path",
                    str(last),
                ]
                with patch.object(sys, "argv", resume_cli):
                    train.main()
                epoch.assert_not_called()
        finally:
            logger = logging.getLogger("jrpp")
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()
