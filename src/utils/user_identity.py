"""Training-author vocabulary and deterministic embedding initialization."""
import copy
import hashlib
import json
import torch

POLICY = 'train-user-vocabulary-v1'
CAPACITY = 100000
IDENTITY_FIELDS = ('policy', 'vocabulary', 'vocabulary_sha256', 'capacity', 'oov_index')


def vocabulary_digest(users):
    return hashlib.sha256(json.dumps(list(users), ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def build_identity(train_users, capacity=CAPACITY):
    if type(capacity) is not int or capacity != CAPACITY:
        raise ValueError('Author vocabulary capacity must be 100000')
    users = sorted(set(str(x) for x in train_users))
    if not users or any(not x for x in users) or len(users) > capacity:
        raise ValueError('Train user vocabulary is empty or exceeds the fixed UID capacity')
    return {'policy': POLICY, 'vocabulary': users, 'vocabulary_sha256': vocabulary_digest(users),
        'capacity': capacity, 'oov_index': 0}


def validate_identity(metadata):
    if not isinstance(metadata, dict) or any(key not in metadata for key in IDENTITY_FIELDS):
        raise ValueError('A complete training-author vocabulary is required')
    users = metadata['vocabulary']
    if not isinstance(users, list) or any(not isinstance(x, str) or not x for x in users):
        raise ValueError('Invalid user vocabulary metadata')
    if type(metadata['oov_index']) is not int or metadata['oov_index'] != 0:
        raise ValueError('User vocabulary OOV index must be zero')
    canonical = build_identity(users, metadata['capacity'])
    if any(metadata[key] != canonical[key] for key in IDENTITY_FIELDS):
        raise ValueError('Corrupt/noncanonical user vocabulary metadata')
    # Additional metadata is provenance, not a switch in model behavior.
    return copy.deepcopy(metadata)


def checkpoint_identity(checkpoint):
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('config'), dict):
        raise ValueError('Checkpoint must contain configuration and a training-author vocabulary')
    saved = validate_identity(checkpoint.get('user_identity'))
    configured = validate_identity(checkpoint['config'].get('user_identity'))
    if saved != configured:
        raise ValueError('Checkpoint user identity metadata differs from its configuration')
    return saved


def apply_identity(split, metadata):
    metadata = validate_identity(metadata)
    lookup = {raw: i+1 for i,raw in enumerate(metadata['vocabulary'])}
    split.user_ids = torch.tensor([lookup.get(str(x), 0) for x in split.original_user_ids], dtype=torch.long)
    split.user_identity = copy.deepcopy(metadata)


def configure_training_identity(train, valid, checkpoint=None):
    if checkpoint is not None:
        metadata = checkpoint_identity(checkpoint)
        if set(metadata['vocabulary']) != set(str(x) for x in train.original_user_ids):
            raise ValueError('Saved vocabulary is not exactly the current train user set')
    else:
        metadata = build_identity(train.original_user_ids)
    apply_identity(train, metadata)
    apply_identity(valid, metadata)
    return metadata


def configure_model_identity(model, metadata):
    from models.mol.mol_query_embeddings import RecoMoLQueryEmbeddingsFn
    metadata = validate_identity(metadata)
    modules = [m for m in model.modules() if isinstance(m, RecoMoLQueryEmbeddingsFn)]
    if len(modules) != 1 or modules[0]._uid_embedding_hash_sizes != [CAPACITY]:
        raise ValueError('Expected a single 100001x256 author embedding table')
    if getattr(modules[0], '_uid_embeddings_0').weight.shape != (CAPACITY+1, 256):
        raise ValueError('Author embedding table capacity/dimension differs from its configuration')
    model.user_identity = metadata


def initialize_user_identity(model):
    from models.mol.mol_query_embeddings import RecoMoLQueryEmbeddingsFn
    from utils.data_contract import encode_ids
    metadata = validate_identity(model.user_identity)
    module = next(m for m in model.modules() if isinstance(m, RecoMoLQueryEmbeddingsFn))
    weight = getattr(module, '_uid_embeddings_0').weight
    codes = encode_ids(metadata['vocabulary'])
    source = torch.tensor([(codes[x] % CAPACITY)+1 for x in metadata['vocabulary']],
        dtype=torch.long, device=weight.device)
    with torch.no_grad():
        # Select from the complete initialized table before writing any row.
        initialized_table = weight.detach().clone()
        weight[1:len(source)+1].copy_(initialized_table.index_select(0, source))
        weight[0].zero_()
    record = {'method': 'identity-derived-row-selection',
        'vocabulary_sha256': metadata['vocabulary_sha256']}
    model.user_embedding_initialization = record
    return record


def restore_identity_to_splits(checkpoint, train, valid):
    return configure_training_identity(train, valid, checkpoint=checkpoint)
