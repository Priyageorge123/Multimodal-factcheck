import json

# Path to your text_only_event.json file
input_path = r'c:\Users\priya\AppData\Local\Temp\fbe23285-7dec-47ef-a301-fc9c9618b224_m2e2_annotations (1).zip.224\m2e2_annotations\text_only_event.json'
output_path = 'event_schema.json'

event_types = {}

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for entry in data:
    for event in entry.get('golden-event-mentions', []):
        event_type = event.get('event_type')
        trigger = event.get('trigger', {}).get('text')
        if not event_type:
            continue
        if event_type not in event_types:
            event_types[event_type] = {
                'triggers': set(),
                'roles': set()
            }
        if trigger:
            event_types[event_type]['triggers'].add(trigger)
        for arg in event.get('arguments', []):
            role = arg.get('role')
            if role:
                event_types[event_type]['roles'].add(role)

# Convert sets to sorted lists for JSON serialization
event_types_json = {
    etype: {
        'triggers': sorted(list(info['triggers'])),
        'roles': sorted(list(info['roles']))
    }
    for etype, info in event_types.items()
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(event_types_json, f, indent=2, ensure_ascii=False)

print(f"Extracted event schema written to {output_path}")