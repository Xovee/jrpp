"""Train-only author inputs, deterministic initialization, gradients and resume."""
from argparse import Namespace
import copy
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'src'))
import train
from models.JRPP import JRPP
from models.initialization import initialize_output_bias_from_train
from models.mol.mol_query_embeddings import RecoMoLQueryEmbeddingsFn
from utils.data_contract import encode_ids
from utils.data_loader import PreparedSplit, collate_popularity_batch
from utils.runtime import capture_random_state, seed_everything
from utils.user_identity import (IDENTITY_FIELDS, build_identity, checkpoint_identity,
    configure_training_identity, initialize_user_identity, validate_identity)


def fixtures():
    generator = np.random.default_rng(77)
    raw = {'image_id': np.asarray(['image-a','image-b','image-c','image-d','image-e','image-f']),
        'user_id': np.asarray(['1','100001','1','100001','1','100001']),
        'label': np.asarray([2.,4.,3.,5.,2.5,4.5], dtype=np.float32),
        'cls_vec': generator.standard_normal((6,768)).astype(np.float32),
        'merged_text_vec': generator.standard_normal((6,768)).astype(np.float32)}
    valid = {k:v[:2].copy() for k,v in raw.items()}
    valid['image_id'] = np.asarray(['query-a','query-b'])
    valid['user_id'] = np.asarray(['1','unseen-validation-author'])
    image_codes = encode_ids([*raw['image_id'], *valid['image_id']])
    protocol = {'fixture': 'train-only-author-inputs'}
    return (PreparedSplit(raw,image_codes,[],protocol),
        PreparedSplit(valid,image_codes,[],protocol))


def configuration():
    config = yaml.safe_load((SOURCE / 'src/config/config.yaml').read_text())
    config['retrieval']['top_k'] = 4
    config['data_protocol'] = {'fixture': 'train-only-author-inputs'}
    return config


def uid_weight(model):
    module = next(x for x in model.modules() if isinstance(x, RecoMoLQueryEmbeddingsFn))
    return module._uid_embeddings_0.weight


def assert_rng_equal(case, a, b):
    case.assertEqual(a['python'], b['python'])
    case.assertEqual(a['numpy'][0], b['numpy'][0])
    np.testing.assert_array_equal(a['numpy'][1], b['numpy'][1])
    case.assertEqual(a['numpy'][2:], b['numpy'][2:])
    case.assertTrue(torch.equal(a['torch'], b['torch']))
    case.assertEqual(len(a['cuda']), len(b['cuda']))
    for x,y in zip(a['cuda'],b['cuda']):
        case.assertTrue(torch.equal(x,y))


def close_logger():
    logger = logging.getLogger('jrpp')
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


class UserIdentityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.args = Namespace(embSize=512, data_name='smpd', dropout=0.2)

    def model(self):
        bank, valid = fixtures()
        metadata = configure_training_identity(bank, valid)
        config = configuration()
        config['user_identity'] = metadata
        seed_everything(12)
        model = JRPP(self.args,config,0,dropout=0.2)
        initialize_output_bias_from_train(model, bank, bank.protocol)
        return model, bank, valid, metadata

    def test_train_only_oov_and_unconfigured_inputs(self):
        bank, valid = fixtures()
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            bank[0]
        metadata = configure_training_identity(bank, valid)
        self.assertEqual(set(metadata), set(IDENTITY_FIELDS))
        self.assertEqual(metadata['vocabulary'], ['1','100001'])
        self.assertEqual(bank.user_ids.tolist(), [1,2,1,2,1,2])
        self.assertEqual(valid.user_ids.tolist(), [1,0])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            build_identity((str(i) for i in range(100001)))

    def test_metadata_integrity_and_opaque_provenance(self):
        metadata = build_identity(['a','b'])
        extended = dict(metadata, provenance={'source':'cpu-fixture'}, note='author inputs')
        self.assertEqual(validate_identity(extended), extended)
        for key, value in (('vocabulary_sha256','wrong'), ('vocabulary',['b','a']),
                           ('vocabulary',['a','a']), ('capacity',99999), ('oov_index',1),
                           ('oov_index',False), ('policy','unknown')):
            with self.subTest(key=key, value=value):
                corrupt = copy.deepcopy(metadata)
                corrupt[key] = value
                with self.assertRaises(ValueError):
                    validate_identity(corrupt)
        for key in IDENTITY_FIELDS:
            incomplete = dict(metadata)
            incomplete.pop(key)
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                validate_identity(incomplete)

    def test_deterministic_initial_rows_non_uid_parameters_and_rng(self):
        model, bank, _, metadata = self.model()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        rng = capture_random_state()
        initialized = initialize_user_identity(model)
        weight_key = next(k for k in before if '_uid_embeddings_0.weight' in k)
        # These two fixed author identifiers select initial row 2 independently.
        for index in (1,2):
            self.assertTrue(torch.equal(uid_weight(model)[index], before[weight_key][2]))
        self.assertEqual(int(torch.count_nonzero(uid_weight(model)[0])),0)
        self.assertTrue(torch.equal(uid_weight(model)[3:], before[weight_key][3:]))
        for key,value in model.state_dict().items():
            if key != weight_key:
                self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(initialized['vocabulary_sha256'],metadata['vocabulary_sha256'])
        assert_rng_equal(self, rng, capture_random_state())

    def test_distinct_author_gradients_and_zero_oov_gradient(self):
        model, bank, _, _ = self.model()
        initialize_user_identity(model)
        model.eval()
        batch = collate_popularity_batch(bank[:2])
        items = train.move_items_to_device(bank,torch.device('cpu'))
        prediction,kl = model(*train._batch_inputs(batch,items))
        loss = (prediction-batch.y).square().mean()+kl
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        one,two = uid_weight(model).grad[1],uid_weight(model).grad[2]
        self.assertTrue(torch.isfinite(one).all() and torch.isfinite(two).all())
        self.assertGreater(float(one.norm()),0.)
        self.assertGreater(float(two.norm()),0.)
        self.assertFalse(torch.equal(one,two))
        self.assertEqual(int(torch.count_nonzero(uid_weight(model).grad[0])),0)

    def test_checkpoint_requires_matching_complete_training_vocabulary(self):
        bank,valid = fixtures()
        metadata = build_identity(bank.original_user_ids)
        checkpoint = {'config':{'user_identity':metadata},'user_identity':metadata}
        configure_training_identity(bank,valid,checkpoint=checkpoint)
        self.assertEqual(valid.user_ids.tolist(),[1,0])
        for incomplete in ({'config':{}}, {'config':{'user_identity':metadata}},
                           {'config':{},'user_identity':metadata}):
            with self.assertRaisesRegex(ValueError,'vocabulary'):
                checkpoint_identity(incomplete)
        inconsistent = copy.deepcopy(checkpoint)
        inconsistent['config']['user_identity'] = copy.deepcopy(inconsistent['config']['user_identity'])
        inconsistent['config']['user_identity']['note'] = 'different'
        with self.assertRaisesRegex(ValueError,'metadata differs'):
            checkpoint_identity(inconsistent)
        other = build_identity(['another-author'])
        with self.assertRaisesRegex(ValueError,'train user set'):
            configure_training_identity(bank,valid,checkpoint={'user_identity':other,
                'config':{'user_identity':other}})
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            JRPP(self.args,configuration(),0)

    def test_native_continuous_resume_matches_without_reinitialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve()
            def run(name, epochs, resume=None):
                bank,valid = fixtures()
                argv=['train.py','--data-name','smpd','--output-dir',str(target),
                    '--device','cpu','--epochs',str(epochs),'--batch-size','2','--patience','0']
                argv += ['--resume-path',str(resume)] if resume else ['--run-name',name]
                with patch.object(train,'read_data',return_value=(bank,valid)), \
                     patch.object(train,'load_config',return_value=configuration()), \
                     patch.object(sys,'argv',argv):
                    if resume is None:
                        train.main()
                    else:
                        with patch.object(train,'initialize_user_identity',side_effect=AssertionError('resume must not reinitialize')):
                            train.main()
                return target/'smpd'/name/'checkpoints/JRPP_last.pt'
            try:
                expected = torch.load(run('continuous',2),map_location='cpu',weights_only=False)
                expected_rng = capture_random_state()
                partial = run('resumed',1)
                actual = torch.load(run('resumed',2,resume=partial),map_location='cpu',weights_only=False)
                self.assertEqual(expected['user_identity'],actual['user_identity'])
                for key,value in expected['model_state_dict'].items():
                    self.assertTrue(torch.equal(value,actual['model_state_dict'][key]),key)
                for key,state in expected['optimizer_state_dict']['state'].items():
                    for field,value in state.items():
                        self.assertTrue(torch.equal(value,actual['optimizer_state_dict']['state'][key][field]))
                assert_rng_equal(self,expected_rng,capture_random_state())
            finally:
                close_logger()


if __name__ == '__main__':
    unittest.main()
