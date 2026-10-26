"""Explicit local-only example. Importing this module never loads a model."""

import argparse
import json
from pathlib import Path

import torch

from neuron_steering import Target, calibrate, discover_layers, steer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", default="steering_config.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    targets = [Target(**item) for item in settings["targets"]]

    # This class must be provided by the user's installed Transformers version.
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32,
    ).to(args.device).eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(args.device)
    layers = discover_layers(model)
    doses = calibrate(model, layers, targets, inputs,
                      skip_first_token=settings["skip_first_token"])
    print(json.dumps([dict(layer=t.layer, neuron=t.neuron, strength=t.strength,
                           dose=d) for t, d in doses.items()], indent=2))
    # A fresh generate call builds its own cache after unsteered calibration.
    with torch.inference_mode(), steer(layers, doses):
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                do_sample=False)
    print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True))


if __name__ == "__main__":
    main()
