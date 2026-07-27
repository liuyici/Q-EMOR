from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import network
import utils
from dataloader import UnalignedDataLoader
from modules import load_seed, load_seed_iv, z_score
from utils import LabelSmooth


LEARNING_RATE = 0.01
BATCH_SIZE = 50
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 1500
CHANNEL_MASK_RATIO = 0.50
ADVERSARIAL_WEIGHT = 0.50
MARGINAL_MMD_WEIGHT = 0.50
GRL_COEFFICIENT = 0.50
NOISE_STD = 0.01

DATASET_SPECS = {
    "seed": {"classes": 3, "channels": 62},
    "seed-iv": {"classes": 4, "channels": 62},
    "faced": {"classes": 9, "channels": 32},
}


class SolverDFN(network.DFN):
    """Executable DFN path used by this solver.

    The repository's DFN exposes the required layers.  This explicit forward
    path removes any ambiguity about flattening and the returned logits.
    """

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = inputs.reshape(inputs.shape[0], -1)
        features = self.feature_extractor(flattened)
        features = self.bottleneck(features)
        logits = self.fc(features)
        return features, logits


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        return -ctx.coefficient * gradients, None


def _device(args) -> torch.device:
    requested = str(getattr(args, "device", "cuda")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _parameter_groups(model: nn.Module):
    return _unwrap(model).get_parameters()


def _log(args, payload: Mapping[str, object]) -> None:
    message = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    print(message)
    log_file = getattr(args, "log_file", None)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def _macro_ovr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Macro one-vs-rest AUC from continuous decision scores.

    A class absent from a particular evaluation fold has no defined binary
    ROC curve and is excluded from that fold's macro average.
    """

    class_auc = []
    for class_index in range(probabilities.shape[1]):
        binary_truth = (y_true == class_index).astype(np.int64)
        if np.unique(binary_truth).size < 2:
            continue
        class_auc.append(
            roc_auc_score(binary_truth, probabilities[:, class_index])
        )
    return float(np.mean(class_auc)) if class_auc else float("nan")


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return accuracy, macro-F1, macro-OVR AUC, confusion matrix, predictions."""

    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities must have shape [n_samples, n_classes]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("probabilities contain NaN or infinite values")

    predictions = scores.argmax(axis=1)
    class_labels = np.arange(scores.shape[1])
    accuracy = float(np.mean(predictions == y_true))
    macro_f1 = float(
        f1_score(
            y_true,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
    )
    macro_auc = _macro_ovr_auc(y_true, scores)
    matrix = confusion_matrix(y_true, predictions, labels=class_labels)
    return accuracy, macro_f1, macro_auc, matrix, predictions


def _evaluation_batches(loader: Iterable):
    for batch in loader:
        if isinstance(batch, Mapping):
            yield batch["Tx"], batch["Ty"]
        elif isinstance(batch, Sequence) and len(batch) >= 2:
            yield batch[0], batch[1]
        else:
            raise TypeError("evaluation batches must provide features and labels")


def evaluate_model(
    loader: Iterable,
    model: nn.Module,
    device: torch.device | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate once after adaptation using softmax scores."""

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    feature_batches = []
    probability_batches = []
    label_batches = []

    with torch.no_grad():
        for inputs, labels in _evaluation_batches(loader):
            inputs = inputs.to(device=device, dtype=torch.float32)
            features, logits = model(inputs)
            feature_batches.append(features.detach().cpu())
            probability_batches.append(F.softmax(logits, dim=1).detach().cpu())
            label_batches.append(labels.detach().cpu().to(torch.int64))

    if not probability_batches:
        raise ValueError("evaluation loader is empty")

    all_features = torch.cat(feature_batches).numpy()
    probabilities = torch.cat(probability_batches).numpy()
    all_labels = torch.cat(label_batches).numpy()
    accuracy, macro_f1, macro_auc, matrix, predictions = classification_metrics(
        all_labels, probabilities
    )
    return accuracy, macro_f1, macro_auc, matrix, all_features, predictions


def test_suda(loader, model):
    """Compatibility wrapper; it performs one evaluation pass only."""

    accuracy, f1, auc, matrix, _, _ = evaluate_model(loader, model)
    return accuracy, f1, auc, matrix


def test_muda(dataset_test, model):
    """Compatibility wrapper for the mapping batches used by legacy loaders."""

    return evaluate_model(dataset_test, model)


def class_conditional_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_logits: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Hard-pseudo-label conditional mean matching from manuscript Eq. (28)."""

    target_pseudo_labels = target_logits.detach().argmax(dim=1)
    losses = []
    for class_index in range(num_classes):
        source_mask = source_labels.reshape(-1) == class_index
        target_mask = target_pseudo_labels == class_index
        # Eq. (28) is undefined for an empty class in a minibatch.  Skipping
        # that class is deterministic and avoids division by zero.
        if not torch.any(source_mask) or not torch.any(target_mask):
            continue
        source_centroid = source_features[source_mask].mean(dim=0)
        target_centroid = target_features[target_mask].mean(dim=0)
        losses.append(torch.sum((source_centroid - target_centroid) ** 2))

    if not losses:
        return (source_features.sum() + target_features.sum()) * 0.0
    return torch.stack(losses).sum()


def _balanced_adversarial_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    discriminator: nn.Module,
) -> torch.Tensor:
    sample_count = min(source_features.shape[0], target_features.shape[0])
    source_balanced = source_features[:sample_count]
    target_balanced = target_features[:sample_count]
    features = torch.cat((source_balanced, target_balanced), dim=0)
    reversed_features = _GradientReverse.apply(features, GRL_COEFFICIENT)
    domain_scores = discriminator(reversed_features).reshape(-1, 1)
    domain_labels = torch.cat(
        (
            torch.zeros((sample_count, 1), device=features.device),
            torch.ones((sample_count, 1), device=features.device),
        ),
        dim=0,
    )
    return F.binary_cross_entropy(domain_scores, domain_labels)


def _complete_channel_mask(
    inputs: torch.Tensor,
    mask_ratio: float,
) -> torch.Tensor:
    """Mask whole channels with one mask shared by all windows/bands."""

    if inputs.ndim < 3:
        raise ValueError(
            "CMAE expects a batch plus channel and frequency dimensions"
        )
    batch_size = inputs.shape[0]
    channel_axis = inputs.ndim - 2
    num_channels = inputs.shape[channel_axis]
    channels_to_mask = int(num_channels * mask_ratio)
    mask = torch.ones_like(inputs)
    for sample_index in range(batch_size):
        selected = torch.randperm(num_channels, device=inputs.device)[
            :channels_to_mask
        ]
        index = [slice(None)] * inputs.ndim
        index[0] = sample_index
        index[channel_axis] = selected
        mask[tuple(index)] = 0.0
    return mask


def _load_faced_npz(file_path: str):
    """Load subject-wise FACED features from a portable NPZ file.

    The archive must contain X and Y.  Each can be an object array with one
    entry per subject or a dense array whose first dimension is the subject.
    """

    path = Path(file_path)
    if path.is_dir():
        path = path / "faced_features.npz"
    with np.load(path, allow_pickle=True) as archive:
        features = archive["X"]
        labels = archive["Y"]
    if len(features) != len(labels):
        raise ValueError("FACED X and Y must contain the same number of subjects")
    return (
        {index: np.asarray(value) for index, value in enumerate(features)},
        {index: np.asarray(value) for index, value in enumerate(labels)},
    )


def _load_dataset(args):
    if args.dataset == "seed":
        return load_seed(args.file_path, session=args.session, feature="de_LDS")
    if args.dataset == "seed-iv":
        if args.mixed_sessions == "per_session":
            return load_seed_iv(args.file_path, session=args.session)
        if args.mixed_sessions != "mixed":
            raise ValueError("mixed_sessions must be 'per_session' or 'mixed'")

        sessions = [
            load_seed_iv(args.file_path, session=session)
            for session in (1, 2, 3)
        ]
        X, Y = {}, {}
        for subject in sessions[0][0]:
            standardized = [z_score(data[0][subject])[0] for data in sessions]
            X[subject] = np.concatenate(standardized, axis=0)
            Y[subject] = np.concatenate(
                [data[1][subject] for data in sessions], axis=0
            )
        return X, Y
    if args.dataset == "faced":
        return _load_faced_npz(args.file_path)
    raise ValueError(f"unsupported dataset: {args.dataset}")


def _target_subject(X: Mapping, target_number: int):
    subject_keys = list(X.keys())
    index = target_number - 1
    if index < 0 or index >= len(subject_keys):
        raise IndexError(
            f"target must be in [1, {len(subject_keys)}], got {target_number}"
        )
    return subject_keys[index], index


def _model_input_size(features: np.ndarray) -> int:
    if features.ndim < 2:
        raise ValueError("features must include sample and feature dimensions")
    return int(np.prod(features.shape[1:]))


def Q_EMO(args):
    """Pre-train Q-EMOR without evaluating target ground-truth labels."""

    if args.dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    device = _device(args)
    num_classes = DATASET_SPECS[args.dataset]["classes"]
    X, Y = _load_dataset(args)
    target_key, target_index = _target_subject(X, args.target)
    target_features = np.asarray(X[target_key])
    target_features, target_mean, target_std = z_score(target_features)
    del target_mean, target_std

    # The pre-training loader receives sentinel labels for the held-out domain.
    # Therefore target ground truth cannot be consumed by any training step.
    unlabeled_target = np.full(len(target_features), -1, dtype=np.int64)
    train_loader = UnalignedDataLoader()
    train_loader.initialize(
        len(X),
        X,
        Y,
        target_features,
        unlabeled_target,
        target_index,
        BATCH_SIZE,
        BATCH_SIZE,
        drop_last_testing=True,
        shuffle_testing=True,
    )
    datasets = train_loader.load_data()

    input_size = _model_input_size(target_features)
    model = SolverDFN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        bottleneck_dim=args.bottleneck_dim,
        class_num=num_classes,
        radius=args.radius,
    ).to(device)
    num_source_domains = len(X) - 1
    decoders = nn.ModuleList(
        [
            network.Decoder(
                hidden_size=args.bottleneck_dim,
                out_dim=input_size,
            )
            for _ in range(num_source_domains)
        ]
    ).to(device)
    discriminator = network.DiscriminatorDANN(
        in_feature=model.output_num(),
        radius=10.0,
        hidden_size=args.bottleneck_dim,
        max_iter=PRETRAIN_EPOCHS,
    ).to(device)

    classifier_groups = [_parameter_groups(model)[2]]
    feature_groups = list(_parameter_groups(model)[:2])
    feature_groups.extend(discriminator.get_parameters())
    for decoder in decoders:
        feature_groups.extend(decoder.get_parameters())

    classifier_optimizer = torch.optim.AdamW(
        classifier_groups, lr=LEARNING_RATE, weight_decay=0.01
    )
    feature_optimizer = torch.optim.AdamW(
        feature_groups, lr=LEARNING_RATE, weight_decay=0.01
    )
    criterion = LabelSmooth(
        num_class=num_classes, device=str(device)
    ).to(device)
    reconstruction_criterion = utils.CosineSimilarityLoss().to(device)
    loss_history = []

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        discriminator.train()
        decoders.train()
        epoch_losses = []

        for data in datasets:
            source_inputs = [
                data[f"Sx{domain + 1}"].to(device=device, dtype=torch.float32)
                for domain in range(num_source_domains)
            ]
            source_labels = [
                data[f"Sy{domain + 1}"].to(device=device, dtype=torch.long)
                for domain in range(num_source_domains)
            ]
            target_inputs = data["Tx"].to(device=device, dtype=torch.float32)
            target_features_batch, _ = model(target_inputs)

            source_feature_batches = []
            source_logit_batches = []
            marginal_mmd_loss = target_features_batch.sum() * 0.0
            reconstruction_loss = target_features_batch.sum() * 0.0

            for domain, clean_source in enumerate(source_inputs):
                noisy_source = clean_source + torch.randn_like(clean_source) * NOISE_STD
                source_features_batch, source_logits_batch = model(noisy_source)
                source_feature_batches.append(source_features_batch)
                source_logit_batches.append(source_logits_batch)

                mask = _complete_channel_mask(
                    noisy_source, CHANNEL_MASK_RATIO
                )
                masked_features, _ = model(noisy_source * mask)
                marginal_mmd_loss = marginal_mmd_loss + utils.marginal(
                    masked_features, target_features_batch
                )

                reconstructed = decoders[domain](masked_features).reshape_as(clean_source)
                masked_positions = 1.0 - mask
                reconstruction_loss = reconstruction_loss + reconstruction_criterion(
                    reconstructed * masked_positions,
                    clean_source * masked_positions,
                )

            marginal_mmd_loss = marginal_mmd_loss / num_source_domains
            reconstruction_loss = reconstruction_loss / num_source_domains
            all_source_features = torch.cat(source_feature_batches, dim=0)
            all_source_logits = torch.cat(source_logit_batches, dim=0)
            all_source_labels = torch.cat(source_labels, dim=0).reshape(-1)

            classification_loss = criterion(
                all_source_logits, all_source_labels
            )
            adversarial_loss = _balanced_adversarial_loss(
                all_source_features, target_features_batch, discriminator
            )
            # Manuscript Eq. (27).
            total_loss = (
                classification_loss
                + ADVERSARIAL_WEIGHT * adversarial_loss
                + MARGINAL_MMD_WEIGHT * marginal_mmd_loss
                + reconstruction_loss
            )

            classifier_optimizer.zero_grad(set_to_none=True)
            feature_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            classifier_optimizer.step()
            feature_optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu()))

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        _log(
            args,
            {
                "phase": "pretrain",
                "epoch": epoch + 1,
                "epochs": PRETRAIN_EPOCHS,
                "training_loss": mean_loss,
            },
        )

    # Target metrics are deliberately unavailable before adaptation.
    unavailable = float("nan")
    empty_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    return (
        X,
        Y,
        unavailable,
        unavailable,
        unavailable,
        empty_matrix,
        model,
        loss_history,
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _fine_tune_unlabeled_target(
    source_loader: DataLoader,
    target_loader: DataLoader,
    model: nn.Module,
    args,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Adapt using source labels and target features only.

    This function cannot access target ground-truth labels by construction.
    """

    classifier_optimizer = torch.optim.AdamW(
        [_parameter_groups(model)[2]],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    feature_optimizer = torch.optim.AdamW(
        list(_parameter_groups(model)[:2]),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    criterion = LabelSmooth(
        num_class=num_classes, device=str(device)
    ).to(device)

    steps_per_epoch = max(len(source_loader), len(target_loader))
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)
        epoch_losses = []

        for _ in range(steps_per_epoch):
            source_batch, source_iterator = _next_batch(
                source_iterator, source_loader
            )
            target_batch, target_iterator = _next_batch(
                target_iterator, target_loader
            )
            source_inputs, source_labels = source_batch
            (target_inputs,) = target_batch
            source_inputs = source_inputs.to(device=device, dtype=torch.float32)
            source_labels = source_labels.to(device=device, dtype=torch.long)
            target_inputs = target_inputs.to(device=device, dtype=torch.float32)

            source_features, source_logits = model(source_inputs)
            target_features, target_logits = model(target_inputs)
            classification_loss = criterion(
                source_logits, source_labels.reshape(-1)
            )
            target_mmd_loss = class_conditional_mmd(
                source_features,
                target_features,
                source_labels,
                target_logits,
                num_classes,
            )
            # Manuscript Eq. (29): no entropy or target-label term.
            total_loss = classification_loss + target_mmd_loss

            classifier_optimizer.zero_grad(set_to_none=True)
            feature_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            feature_optimizer.step()
            classifier_optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu()))

        _log(
            args,
            {
                "phase": "finetune",
                "epoch": epoch + 1,
                "epochs": FINETUNE_EPOCHS,
                "training_loss": float(np.mean(epoch_losses)),
                "checkpoint_rule": "final_epoch",
            },
        )
    return model


def FunEing(X, Y, model, args):
    """Fine-tune without target labels, then evaluate the final epoch once."""

    if args.dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    device = _device(args)
    num_classes = DATASET_SPECS[args.dataset]["classes"]
    target_key, _ = _target_subject(X, args.target)

    source_features = []
    source_labels = []
    for subject in X:
        if subject == target_key:
            continue
        standardized, _, _ = z_score(np.asarray(X[subject]))
        source_features.append(standardized)
        source_labels.append(np.asarray(Y[subject]))

    Sx = np.concatenate(source_features, axis=0)
    Sy = np.concatenate(source_labels, axis=0)
    Tx, _, _ = z_score(np.asarray(X[target_key]))

    source_dataset = TensorDataset(
        torch.as_tensor(Sx, dtype=torch.float32),
        torch.as_tensor(Sy, dtype=torch.long),
    )
    target_adaptation_dataset = TensorDataset(
        torch.as_tensor(Tx, dtype=torch.float32)
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_adaptation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = model.to(device)
    model = _fine_tune_unlabeled_target(
        source_loader,
        target_loader,
        model,
        args,
        num_classes,
        device,
    )

    # Ground-truth target labels enter the execution path only here, after all
    # optimizer steps.  The predetermined final epoch is evaluated once.
    Ty = np.asarray(Y[target_key])
    evaluation_dataset = TensorDataset(
        torch.as_tensor(Tx, dtype=torch.float32),
        torch.as_tensor(Ty, dtype=torch.long),
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    final_acc, final_f1, final_auc, final_mat, _, _ = evaluate_model(
        evaluation_loader, model, device
    )
    _log(
        args,
        {
            "phase": "final_evaluation",
            "checkpoint_rule": "final_epoch",
            "accuracy": final_acc,
            "macro_f1": final_f1,
            "macro_ovr_auc": final_auc,
        },
    )
    return final_acc, final_f1, final_auc, final_mat, model

