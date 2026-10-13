"""The fresh random stream is reset once; resume restores the saved stream."""
import logging
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import train
from utils.runtime import seed_everything
from utils.user_identity import build_identity, validate_identity


class Split(list):
    protocol = {'fixture':'stochastic-training-state'}
    original_user_ids = ['author-a']


class SmallModel(nn.Sequential):
    def __init__(self,args,config,meta_dim,dropout):
        super().__init__(nn.Linear(2,4),nn.Dropout(dropout),nn.Linear(4,1))
        self.config = config
        self.user_identity = validate_identity(config['user_identity'])

    @property
    def regressor(self):
        return self


def random_sample():
    return random.random(),float(np.random.random()),torch.rand(4)


class TrainingRandomnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.first_batch_state = None

    def train_epoch(self,model,loader,optimizer,criterion,items,device,ib_weight,epoch):
        model.train()
        sample = random_sample()
        if self.first_batch_state is None:
            self.first_batch_state = ({k:v.clone() for k,v in model.state_dict().items()},sample)
        x = torch.arange(24,dtype=torch.float32).reshape(12,2)/24
        for (batch,) in DataLoader(TensorDataset(x),batch_size=3,shuffle=True):
            optimizer.zero_grad()
            target = random.random()+np.random.random()+torch.rand(())
            loss = (model(batch)-target).square().mean()
            loss.backward()
            optimizer.step()
        return {'loss':float(loss.detach()),'mse_loss':float(loss.detach())}

    def run_main(self,name,epochs,seed=12,resume=None):
        split = Split([Namespace(meta_features=torch.empty(0),y=torch.tensor([3.25]))])
        valid = Split([Namespace(meta_features=torch.empty(0),y=torch.tensor([999.]))])
        argv=['train.py','--output-dir',self.temp.name,'--device','cpu',
            '--epochs',str(epochs),'--seed',str(seed),'--patience','0']
        argv += ['--resume-path',str(resume)] if resume else ['--run-name',name]
        try:
            with ExitStack() as stack:
                for attribute,value in (('load_config',{}),('meta_fields_for',[]),
                    ('read_data',(split,valid)),('move_items_to_device',())):
                    stack.enter_context(patch.object(train,attribute,return_value=value))
                stack.enter_context(patch.object(train,'JRPP',SmallModel))
                stack.enter_context(patch.object(train,'initialize_user_identity',return_value=None))
                if resume:
                    stack.enter_context(patch.object(train,'initialize_output_bias_from_train',
                        side_effect=AssertionError('resume must not reset bias')))
                stack.enter_context(patch.object(train,'train_one_epoch',side_effect=self.train_epoch))
                stack.enter_context(patch.object(train,'evaluate',return_value={'mse':1.,'mae':1.,'src':0.}))
                stack.enter_context(patch.object(sys,'argv',argv))
                train.main()
            return Path(self.temp.name)/'icip'/name/'checkpoints/JRPP_last.pt'
        finally:
            logger=logging.getLogger('jrpp')
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()

    def assert_sample_equal(self,a,b):
        self.assertEqual(a[:2],b[:2])
        torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)

    def test_native_fresh_reseeds_after_parameter_initialization(self):
        for seed in (12,24,36):
            with self.subTest(seed=seed):
                args=Namespace(dropout=0.2)
                config={'user_identity':build_identity(['author-a'])}
                seed_everything(seed)
                model=SmallModel(args,config,0,0.2)
                train.initialize_output_bias_from_train(model,
                    Split([Namespace(y=torch.tensor([3.25]))]),Split.protocol)
                expected={k:v.clone() for k,v in model.state_dict().items()}
                seed_everything(seed)
                expected_random=random_sample()
                self.first_batch_state=None
                self.run_main(f'fresh-{seed}',1,seed=seed)
                state,sample=self.first_batch_state
                for key in state:
                    torch.testing.assert_close(state[key],expected[key],rtol=0,atol=0)
                self.assert_sample_equal(sample,expected_random)

    def test_native_resume_keeps_saved_random_stream(self):
        expected=torch.load(self.run_main('continuous',2),weights_only=False)
        expected_random=random_sample()
        partial=self.run_main('resumed',1)
        seed_everything(9876)
        actual=torch.load(self.run_main('resumed',2,resume=partial),weights_only=False)
        for key,value in actual['model_state_dict'].items():
            torch.testing.assert_close(value,expected['model_state_dict'][key],rtol=0,atol=0)
        self.assert_sample_equal(random_sample(),expected_random)
        self.assertEqual(actual['fresh_rng_policy'],train.FRESH_RNG_POLICY)
        self.assertEqual(actual['output_bias_initialization'],expected['output_bias_initialization'])


if __name__ == '__main__':
    unittest.main()
