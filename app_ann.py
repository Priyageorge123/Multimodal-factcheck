from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import json
import spacy
from spacy.tokens import Doc, Span
from spacy import displacy
from collections import defaultdict

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")  

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

COLORS = {
    "PER": "#a6e22d", "ORG": "#ff7f0e", "LOC": "#1f77b4", "FAC": "#d62728","GPE":"#FFFF00",
    "PRODUCT": "#9467bd", "EVENT": "#bcbd22", "WORK_OF_ART": "#17becf",
    "LAW": "#8c564b", "LANGUAGE": "#e377c2", "DATE": "#2ca02c", "TIME": "#1c9099",
    "PERCENT": "#7f7f7f", "MONEY": "#ff1493", "QUANTITY": "#aec7e8",
    "ORDINAL": "#ffbb78", "CARDINAL": "#98df8a", "NORP": "#c49c94"
}
EVENT_TYPE_COLORS = {
    "Life:Die": "#e6194B",
    "Movement:Transport": "#3cb44b",
    "Transaction:TransferMoney": "#ffe119",
    "Conflict:Attack": "#4363d8",
    "Conflict:Demonstrate": "#f58231",
    "Contact:Meet": "#911eb4",
    "Contact:Phone-Write": "#46f0f0",
    "Justice:ArrestJail": "#f032e6",
}


def extract_events(sentences, event_schema):
    from collections import defaultdict

    event_map = defaultdict(list)

    for sent in sentences:
        full_id = sent.get("sentence_id", "")
        sentence_id = full_id.split("_")[-1] if full_id else ""

        for event in sent.get("golden-event-mentions", []):
            event_type = event.get("event_type")
            trigger = event.get("trigger", {}).get("text")

            # Map actual roles from this sentence
            found_roles = {arg["role"]: arg["text"] for arg in event.get("arguments", [])}

            # Get expected roles from the schema
            expected_roles = event_schema.get(event_type, {}).get("roles", [])

            # Fill all expected roles, use "None" if not found
            role_details = []
            for role in expected_roles:
                role_details.append({
                    "role": role,
                    "text": found_roles.get(role, "None")
                })

            event_map[event_type].append({
                "sentence_id": sentence_id,
                "trigger": trigger,
                "arguments": role_details
            })

    return event_map


def create_docs(sentences):
    words = []
    spaces = []
    entity_spans = []
    event_spans = []
    event_map = {}
    entity_map = set()
    event_types = set()
    dep_labels_all=[]
    dep_docs = []

    offset = 0  

    for sent in sentences:
        sent_words = sent["words"]
        words.extend(sent_words)
        spaces.extend([True] * (len(sent_words) - 1) + [True])  # Ensure spacing between sentences

        for ent in sent.get("golden-entity-mentions", []):
            start = offset + ent["start"]
            end = offset + ent["end"]
            label = ent["entity-type"]
            entity_spans.append((start, end, label))
            entity_map.add(label)
        for event in sent.get("golden-event-mentions", []):
            if "trigger" in event and event["trigger"]:
                start = offset + event["trigger"]["start"]
                end = offset + event["trigger"]["end"]
                label = event["event_type"]
                event_spans.append((start, end, label))
                event_map.setdefault(label, set()).add(event["trigger"]["text"])

        # Dependency parsing for each sentence
        dep_doc = nlp(" ".join(sent_words))
        dep_docs.append(dep_doc)
        dep_labels = sorted(set(token.dep_ for token in dep_doc))
        dep_labels_all.append(dep_labels)

        offset += len(sent_words)

    # Create a full Doc with exact words and spaces
    main_doc = Doc(nlp.vocab, words=words, spaces=spaces)
    ner_doc = main_doc.copy()
    event_doc = main_doc.copy()

    # Assign entity spans
    ner_doc.ents = [Span(main_doc, start, end, label=label) for start, end, label in entity_spans]
    event_doc.ents = [Span(main_doc, start, end, label=label) for start, end, label in event_spans]

    return ner_doc, event_doc, dep_docs, entity_map, dep_labels_all, event_types


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract article_id from filename
            article_id = '.'.join(filename.split('.')[:4])
            return redirect(url_for('view_article', article_id=article_id))

    return render_template("upload.html")

@app.route("/view")
def view_article():
    article_id = request.args.get("article_id")
    if not article_id:
        return "Please provide an article_id", 400

    with open("text_only_event.json", encoding="utf-8") as f:
        data = json.load(f)
    with open("event_schema.json", encoding="utf-8") as f:
        event_schema = json.load(f)

    relevant = [d for d in data if d.get("sentence_id", "").startswith(article_id)]
    if not relevant:
        return "Article ID not found", 404
    
    events = extract_events(relevant,event_schema)

    ner_doc, event_doc, dep_docs, entity_map, dep_labels_all, event_types = create_docs(relevant)

    ner_html = displacy.render(ner_doc, style="ent", options={"colors": COLORS}, page=True)
    event_html = displacy.render(event_doc, style="ent", options={"colors": COLORS}, page=True)
    dep_html_list = [displacy.render(dep, style="dep", page=False) for dep in dep_docs]
    zipped_deps = list(zip(dep_html_list, dep_labels_all))
    return render_template("full_view.html",
                           article_id=article_id,
                           ner_html=ner_html,
                           event_html=event_html,
                           dep_html_list=dep_html_list,
                           entity_map=entity_map,
                           zipped_deps=zipped_deps,
                           event_types=event_types,
                           events=events,
                           event_type_colors=EVENT_TYPE_COLORS,
                           color_map=COLORS)

if __name__ == "__main__":
    app.run(debug=True)
