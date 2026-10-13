"""Only the final bias is initialized from complete finite training labels."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from models.initialization import initialize_output_bias_from_train, OUTPUT_BIAS_POLICY
from train import save_checkpoint, FRESH_RNG_POLICY
from utils.parsers import build_parser
from utils.runtime import seed_everything, capture_random_state, build_optimizer
from utils.user_identity import build_identity
from test_user_identity import assert_rng_equal


class Labels(list):
    def __init__(self, values):
        self.labels = torch.tensor(values,dtype=torch.float32).reshape(-1,1)
        super().__init__([SimpleNamespace(y=value) for value in self.labels])


class Model(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2,8),nn.Dropout(0.2),nn.Linear(8,1))
        self.user_identity = build_identity(['author-a'])
        self.config = {'data_protocol':{'fixture':'finite-train-labels'},
            'user_identity':self.user_identity}

    @property
    def regressor(self):
        return self


class OutputBiasInitializationTests(unittest.TestCase):
    def test_only_final_bias_changes_and_rng_is_preserved(self):
        labels = Labels([1.,0.,0.])
        mean = float(labels.labels.numpy().astype(np.float64).mean())
        for seed in (12,24,36):
            with self.subTest(seed=seed):
                seed_everything(seed)
                model = Model()
                before = {key:value.clone() for key,value in model.state_dict().items()}
                rng = capture_random_state()
                initialize_output_bias_from_train(model,labels,model.config['data_protocol'])
                assert_rng_equal(self,rng,capture_random_state())
                for key,value in model.state_dict().items():
                    expected = torch.full_like(value,mean) if key == '2.bias' else before[key]
                    torch.testing.assert_close(value,expected,rtol=0,atol=0)
                self.assertEqual(model.initialization_policy,OUTPUT_BIAS_POLICY)
                self.assertEqual(model.output_bias_initialization['train_count'],3)
                self.assertEqual(model.output_bias_initialization['actual_bias'],float(torch.tensor(mean,dtype=torch.float32)))

    def test_train_only_mean_and_config_are_preserved(self):
        model = Model()
        before = copy.deepcopy(model.config)
        initialize_output_bias_from_train(model,Labels([2.,4.,8.]),model.config['data_protocol'])
        self.assertEqual(model.output_bias_initialization['train_mean_float64'],14./3.)
        self.assertEqual(model.config,before)
        self.assertEqual(model.output_bias_initialization['parameter_dtype'],'torch.float32')

    def test_empty_or_nonfinite_labels_fail_before_bias_write(self):
        for values in ([],[float('nan')],[float('inf')]):
            with self.subTest(values=values):
                model = Model()
                before = model.regressor[-1].bias.detach().clone()
                with self.assertRaises(ValueError):
                    initialize_output_bias_from_train(model,Labels(values),model.config['data_protocol'])
                torch.testing.assert_close(model.regressor[-1].bias,before,rtol=0,atol=0)
                self.assertFalse(hasattr(model,'output_bias_initialization'))

    def test_checkpoint_saves_current_initialization_and_configuration(self):
        model = Model()
        args = build_parser().parse_args(['--device','cpu'])
        initialize_output_bias_from_train(model,Labels([3.,4.]),model.config['data_protocol'])
        model.fresh_rng_policy = FRESH_RNG_POLICY
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            save_checkpoint(path,model,build_optimizer(model,model.config),0,1.,args,model.config,{})
            saved = torch.load(path,weights_only=False)
            self.assertEqual(saved['initialization_policy'],OUTPUT_BIAS_POLICY)
            self.assertEqual(saved['output_bias_initialization'],model.output_bias_initialization)
            self.assertEqual(saved['config'],model.config)
            self.assertEqual(saved['user_identity'],model.user_identity)


if __name__ == '__main__':
    unittest.main()
