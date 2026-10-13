"""Native training and evaluation use the saved training-author vocabulary."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import train
from models.JRPP import JRPP
from models.initialization import initialize_output_bias_from_train
from utils.parsers import build_parser
from utils.runtime import capture_random_state, seed_everything
from utils.user_identity import configure_training_identity, initialize_user_identity
from test_user_identity import fixtures, configuration, assert_rng_equal, close_logger

spec = importlib.util.spec_from_file_location('jrpp_native_evaluation', ROOT / 'src/test.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_native_fresh_initialization_and_post_initialization_random_state(self):
        bank,valid = fixtures()
        metadata = configure_training_identity(bank,valid)
        args = build_parser().parse_args(['--data-name','smpd','--device','cpu','--batch-size','2'])
        config = configuration()
        config['user_identity'] = metadata
        seed_everything(args.seed)
        model = JRPP(args,config,0,dropout=args.dropout)
        initialize_output_bias_from_train(model,bank,bank.protocol)
        initialize_user_identity(model)
        expected_hashes = {k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()
            for k,v in model.state_dict().items()}
        seed_everything(args.seed)
        expected_rng = capture_random_state()
        del model
        with tempfile.TemporaryDirectory() as temporary:
            bank,valid = fixtures()
            captured = []
            def first_batch(model,*unused):
                captured.append((copy.deepcopy(model.config),capture_random_state(),
                    {k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()
                     for k,v in model.state_dict().items()}))
                return {'loss':1.,'mse_loss':1.}
            argv = ['train.py','--data-name','smpd','--output-dir',temporary,
                '--run-name','current','--device','cpu','--epochs','1','--batch-size','2','--patience','0']
            try:
                with patch.object(train,'read_data',return_value=(bank,valid)), \
                     patch.object(train,'load_config',return_value=configuration()), \
                     patch.object(train,'train_one_epoch',side_effect=first_batch), \
                     patch.object(sys,'argv',argv):
                    train.main()
            finally:
                close_logger()
            actual_config,actual_rng,actual_hashes = captured[0]
            self.assertEqual(actual_config,config)
            self.assertEqual(actual_hashes,expected_hashes)
            assert_rng_equal(self,actual_rng,expected_rng)
            self.assertEqual(sorted(p.name for p in Path(temporary).rglob('*.pt')),
                ['JRPP_best.pt','JRPP_last.pt'])

    def test_native_evaluation_restores_vocabulary_oov_and_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            bank,valid = fixtures()
            args = build_parser().parse_args(['--data-name','smpd','--device','cpu','--batch-size','2'])
            config = configuration()
            metadata = configure_training_identity(bank,valid)
            metadata['provenance'] = {'source':'cpu-fixture'}
            config['user_identity'] = metadata
            seed_everything(12)
            model = JRPP(args,config,0,dropout=0.2)
            initialize_output_bias_from_train(model,bank,bank.protocol)
            initialize_user_identity(model)
            model.eval()
            items = train.move_items_to_device(bank,torch.device('cpu'))
            with torch.no_grad():
                expected,_ = model(*train._batch_inputs(train.collate_popularity_batch(valid),items))
            checkpoint = target/'best.pt'
            payload = {'args':vars(args),'config':config,'user_identity':metadata,
                'model_state_dict':model.state_dict(),'epoch':0}
            torch.save(payload,checkpoint)
            restored_bank,restored_valid = fixtures()
            argv = ['test.py','--model-path',str(checkpoint),'--output-dir',str(target/'evaluation'),
                '--device','cpu','--batch-size','2','--no-tta']
            try:
                with patch.object(evaluation,'read_data',return_value=(restored_bank,restored_valid)), \
                     patch.object(sys,'argv',argv):
                    evaluation.main()
            finally:
                close_logger()
            actual = np.load(target/'evaluation/smpd/test_predictions_plain.npy',allow_pickle=False)
            np.testing.assert_array_equal(actual.reshape(-1),expected.numpy().reshape(-1))
            self.assertEqual(restored_valid.user_ids.tolist(),[1,0])
            self.assertEqual(restored_bank.user_identity,metadata)

    def test_native_evaluation_rejects_missing_vocabulary_before_loading_data(self):
        args = evaluation.build_parser(require_model_path=True).parse_args(['--model-path','unused.pt'])
        saved_args = build_parser().parse_args(['--data-name','smpd'])
        with patch.object(evaluation,'read_data') as read:
            with self.assertRaisesRegex(ValueError,'vocabulary'):
                evaluation.restore_evaluation_settings(args,{'args':vars(saved_args),'config':configuration()})
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
