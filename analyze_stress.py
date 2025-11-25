import json

# Load stress set
stress = [json.loads(l) for l in open('data/stress.jsonl', 'r', encoding='utf-8') if l.strip()]

# Find CREDIT_CARD examples
cc_examples = [s for s in stress if any(e['label'] == 'CREDIT_CARD' for e in s.get('entities', []))]
print("CREDIT_CARD examples from stress set:")
print("="*80)
for i, ex in enumerate(cc_examples[:3]):
    print(f"\n{i+1}. ID: {ex['id']}")
    print(f"   Text: {ex['text']}")
    cc_entities = [e for e in ex['entities'] if e['label'] == 'CREDIT_CARD']
    for e in cc_entities:
        print(f"   Entity: '{ex['text'][e['start']:e['end']]}' [{e['start']}:{e['end']}]")

print("\n" + "="*80)
print("\nEMAIL examples from stress set:")
print("="*80)

email_examples = [s for s in stress if any(e['label'] == 'EMAIL' for e in s.get('entities', []))]
for i, ex in enumerate(email_examples[:3]):
    print(f"\n{i+1}. ID: {ex['id']}")
    print(f"   Text: {ex['text']}")
    email_entities = [e for e in ex['entities'] if e['label'] == 'EMAIL']
    for e in email_entities:
        print(f"   Entity: '{ex['text'][e['start']:e['end']]}' [{e['start']}:{e['end']}]")
