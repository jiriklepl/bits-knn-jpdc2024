"""Capture real causal-model tensors on CPU; imported only by the exporter."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re

from application_inputs import (
    GRADIENT_SEMANTICS,
    SAMPLING_SEMANTICS,
    load_application_manifest,
    sha256_file,
)

DEFAULT_MODEL = "distilbert/distilgpt2"
# https://huggingface.co/distilbert/distilgpt2/tree/2290a62682d06624634c1f46a6ad5be0f47f38aa
DEFAULT_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DEFAULT_PARAMETER = "transformer.h.0.mlp.c_fc.weight"
GRADIENT_LOSS = "mean causal next-token cross entropy; padding labels ignored (-100)"
DEFAULT_PROMPTS = [
    "The morning train arrived at the station just as the rain began.",
    "To prepare the vegetable soup, first wash and chop the carrots.",
    "A telescope collects light from distant stars and galaxies.",
    "The programmer opened the source file and found a missing condition.",
    "Beyond the old bridge, a narrow path led through the forest.",
    "The library keeps a record of every book borrowed by its readers.",
    "When water freezes, its molecules form an ordered crystal structure.",
    "After the final whistle, the players shook hands and left the field.",
]


def validate_capture_options(model, revision, prompts, seed, threads):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty Hugging Face model identifier")
    if revision is None:
        if model != DEFAULT_MODEL:
            raise ValueError(
                "A custom model requires --revision with a full commit SHA"
            )
        revision = DEFAULT_REVISION
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full, immutable, lowercase commit SHA")
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise ValueError("prompts must be a nonempty JSON list of nonempty strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be a uint64 integer")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    return revision


def export_model_input(
    operator,
    destination,
    *,
    model=DEFAULT_MODEL,
    revision=None,
    prompts=None,
    parameter=None,
    seed=42,
    threads=1,
    cache_directory=None,
    microbatch_size=8,
    prompt_origin=None,
    local_files_only=False,
):
    """Export last-token logits or the unsliced gradient of one named parameter."""
    if operator not in {"token-sampling", "gradient-compression"}:
        raise ValueError("Unsupported model application")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("The export destination must not already exist")
    prompts = list(DEFAULT_PROMPTS) if prompts is None else prompts
    revision = validate_capture_options(model, revision, prompts, seed, threads)
    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size <= 0
    ):
        raise ValueError("microbatch_size must be a positive integer")
    if prompt_origin is not None and (
        not isinstance(prompt_origin, str) or not prompt_origin.strip()
    ):
        raise ValueError("prompt_origin must be a nonempty string")
    if operator == "token-sampling" and parameter is not None:
        raise ValueError("parameter only applies to gradient-compression")
    parameter = DEFAULT_PARAMETER if parameter is None else parameter
    if not isinstance(parameter, str) or not parameter:
        raise ValueError("parameter must name one complete model parameter")

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    snapshot = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=cache_directory,
            local_files_only=local_files_only,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    network, loading = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        loading.get(key) for key in ["missing_keys", "mismatched_keys", "error_msgs"]
    ):
        raise ValueError(
            "Model weights must load completely without initialized replacements"
        )
    network.to(device="cpu", dtype=torch.float32)
    network.eval()
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    lengths = attention_mask.sum(dim=1)
    minimum_length = 2 if operator == "gradient-compression" else 1
    if torch.any(lengths < minimum_length):
        raise ValueError(f"Each prompt must contain at least {minimum_length} tokens")
    max_positions = getattr(network.config, "max_position_embeddings", None)
    if max_positions is not None and input_ids.shape[1] > max_positions:
        raise ValueError(
            "Prompt exceeds the model context; prompts are never truncated"
        )
    token_data = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
    }
    source = {
        "kind": "captured-causal-language-model",
        "model": model,
        "revision": revision,
        "model_files_sha256": {
            str(path.relative_to(snapshot)): sha256_file(path)
            for path in sorted(snapshot.rglob("*"))
            if path.is_file()
        },
        "prompts": prompts,
        "prompt_origin": prompt_origin
        or (
            "fixed benchmark sentences"
            if prompts == DEFAULT_PROMPTS
            else "user-supplied"
        ),
        **token_data,
        "input_tokens_sha256": hashlib.sha256(
            json.dumps(token_data, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "seed": seed,
        "capture": {
            "device": "cpu",
            "dtype": "float32",
            "mode": "eval",
            "dropout": "disabled",
            "attention_implementation": "eager",
            "threads": threads,
            "deterministic_algorithms": True,
            "use_cache": False,
            "padding_side": "right",
            "padding_token_id": tokenizer.pad_token_id,
            "add_special_tokens": False,
            "truncation": False,
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "numpy",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "huggingface-hub",
                ]
            },
        },
        "exporter_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in [
                "application_inputs.py",
                "model_application_inputs.py",
                "export-application-inputs.py",
            ]
        },
    }
    manifest = {"version": 1, "operator": operator, "source": source, "columns": {}}
    if operator == "token-sampling":
        values = np.empty((len(prompts), network.config.vocab_size), dtype="<f4")
        with torch.no_grad():
            for begin in range(0, len(prompts), microbatch_size):
                end = min(begin + microbatch_size, len(prompts))
                logits = network(
                    input_ids=input_ids[begin:end],
                    attention_mask=attention_mask[begin:end],
                    use_cache=False,
                ).logits
                values[begin:end] = (
                    logits[torch.arange(end - begin), lengths[begin:end] - 1, :]
                    .contiguous()
                    .numpy()
                )
                del logits
        source["capture"].update(
            position="last nonpadding token of each prompt",
            microbatch_size=min(microbatch_size, len(prompts)),
            forward_passes=(len(prompts) + microbatch_size - 1) // microbatch_size,
        )
        manifest.update(
            batch_size=values.shape[0],
            vocabulary_size=values.shape[1],
            semantics=SAMPLING_SEMANTICS,
        )
        column_name = "logits"
    else:
        parameters = dict(network.named_parameters())
        if parameter not in parameters:
            raise ValueError(f"Model has no parameter named {parameter!r}")
        chosen = parameters[parameter]
        if not 0 < chosen.numel() <= 2**31 - 1:
            raise ValueError(
                "The complete parameter must have a positive int32 element count"
            )
        # Freezing other weights leaves this partial derivative unchanged.
        for name, value in parameters.items():
            value.requires_grad_(name == parameter)
        network.zero_grad(set_to_none=True)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        loss = network(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite causal language-model loss")
        loss.backward()
        if chosen.grad is None:
            raise ValueError("The selected parameter did not receive a gradient")
        values = chosen.grad.detach().contiguous().numpy()
        source.update(
            parameter=parameter,
            loss=GRADIENT_LOSS,
            loss_value=float(loss.detach()),
            loss_target_tokens=int((lengths - 1).sum()),
        )
        source["capture"].update(
            trainable_parameters=[parameter],
            backward_passes=1,
            optimizer_steps=0,
            tensor_selection="complete named parameter; no slicing or concatenation",
        )
        manifest.update(
            elements=values.size,
            tensor_shape=list(values.shape),
            semantics=GRADIENT_SEMANTICS,
        )
        column_name = "gradient"
    values = np.asarray(values, dtype="<f4", order="C")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite captured values in {column_name}")
    destination.mkdir(parents=True)
    array_path = destination / f"{column_name}.f32"
    values.tofile(array_path)
    manifest["columns"][column_name] = {
        "file": array_path.name,
        "dtype": "float32",
        "byte_order": "little",
        "shape": list(values.shape),
        "sha256": sha256_file(array_path),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    load_application_manifest(manifest_path)
    return manifest_path
