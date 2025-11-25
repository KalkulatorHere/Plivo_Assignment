import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids, confidences=None, threshold=0.0):
    """Convert BIO tags to spans with improved edge case handling and confidence filtering"""
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_conf = []

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        # Skip special tokens (CLS, SEP, PAD)
        if start == 0 and end == 0:
            continue
        
        conf = confidences[idx] if confidences is not None else 1.0
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            # End current span if exists
            if current_label is not None:
                avg_conf = np.mean(current_conf) if current_conf else 0.0
                if avg_conf >= threshold:
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_conf = []
            continue

        # Split B-/I- prefix from entity type
        if "-" in label:
            prefix, ent_type = label.split("-", 1)
        else:
            continue
            
        if prefix == "B":
            # Save previous span
            if current_label is not None:
                avg_conf = np.mean(current_conf) if current_conf else 0.0
                if avg_conf >= threshold:
                    spans.append((current_start, current_end, current_label))
            # Start new span
            current_label = ent_type
            current_start = start
            current_end = end
            current_conf = [conf]
        elif prefix == "I":
            if current_label == ent_type:
                # Continue current span
                current_end = end
                current_conf.append(conf)
            else:
                # I- without matching B- or different type: start new span
                if current_label is not None:
                    avg_conf = np.mean(current_conf) if current_conf else 0.0
                    if avg_conf >= threshold:
                        spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
                current_conf = [conf]

    # Don't forget the last span
    if current_label is not None:
        avg_conf = np.mean(current_conf) if current_conf else 0.0
        if avg_conf >= threshold:
            spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--confidence_threshold", type=float, default=0.0)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                probs = torch.softmax(logits, dim=-1)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                confidences = probs.max(dim=-1).values.cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids, confidences, args.confidence_threshold)
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()

