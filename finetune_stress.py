"""
Fine-tune on stress set to improve adversarial performance.
Uses a small portion of stress set for few-shot adaptation.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

from src.dataset import PIIDataset, collate_batch
from src.labels import LABELS, ID2LABEL


def evaluate(model, dev_dl, device):
    """Simple validation to track progress"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dev_dl:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
            preds = outputs.logits.argmax(dim=-1)
            for pred, label, mask in zip(preds, labels, attention_mask):
                pred = pred[mask == 1].cpu().tolist()
                label = label[mask == 1].cpu().tolist()
                all_preds.extend([ID2LABEL.get(p, "O") for p in pred])
                all_labels.extend([ID2LABEL.get(l, "O") for l in label if l != -100])
    
    model.train()
    avg_loss = total_loss / max(1, len(dev_dl))
    
    # Simple accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / max(1, len(all_labels))
    
    return avg_loss, accuracy


def main():
    # Use a subset of stress for fine-tuning (50 examples)
    # Keep other 50 for evaluation
    model_dir = "out"
    stress_path = "data/stress.jsonl"
    out_dir = "out_stress_ft"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model and tokenizer from out/...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    
    # Create train/val split from stress set
    import json
    stress_data = [json.loads(l) for l in open(stress_path, 'r', encoding='utf-8') if l.strip()]
    
    # Use first 60 for training, last 40 for validation
    train_stress = stress_data[:60]
    val_stress = stress_data[60:]
    
    # Save temporary files
    os.makedirs("temp", exist_ok=True)
    with open("temp/stress_train.jsonl", "w", encoding="utf-8") as f:
        for item in train_stress:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open("temp/stress_val.jsonl", "w", encoding="utf-8") as f:
        for item in val_stress:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Fine-tuning on {len(train_stress)} stress examples...")
    print(f"Validating on {len(val_stress)} stress examples...")
    
    train_ds = PIIDataset("temp/stress_train.jsonl", tokenizer, LABELS, max_length=256, is_train=True)
    val_ds = PIIDataset("temp/stress_val.jsonl", tokenizer, LABELS, max_length=256, is_train=True)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=4,  # Smaller batch for fine-tuning
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    model.train()
    
    # Use smaller learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    best_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / max(1, len(train_dl))
        val_loss, val_acc = evaluate(model, val_dl, device)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(out_dir, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  → Saved new best model (acc={val_acc:.4f})")
    
    print(f"\nFine-tuning complete. Best val accuracy: {best_acc:.4f}")
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
