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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

COLORS = {
    "PER": "#a6e22d", "ORG": "#ff7f0e", "LOC": "#1f77b4", "FAC": "#d62728", "GPE": "#FFFF00",
    "PRODUCT": "#9467bd", "EVENT": "#bcbd22", "WORK_OF_ART": "#17becf", "LAW": "#8c564b",
    "LANGUAGE": "#e377c2", "DATE": "#2ca02c", "TIME": "#1c9099", "PERCENT": "#7f7f7f",
    "MONEY": "#ff1493", "QUANTITY": "#aec7e8", "ORDINAL": "#ffbb78", "CARDINAL": "#98df8a",
    "NORP": "#c49c94"
}

EVENT_TYPE_COLORS = {
    "Life:Die": "#e6194B",
    "Movement:Transport": "#3cb44b",
    "Transaction:Transfer-Money": "#ffe119",
    "Conflict:Attack": "#4363d8",
    "Conflict:Demonstrate": "#f58231",
    "Contact:Meet": "#911eb4",
    "Contact:Phone-Write": "#46f0f0",
    "Justice:Arrest-Jail": "#f032e6",
}

def extract_events(sentences, event_schema):
    event_map = defaultdict(list)
    sentence_event_map = defaultdict(set)

    for sent in sentences:
        full_id = sent.get("sentence_id", "")
        sentence_id = full_id.split("_")[-1] if full_id else ""

        for event in sent.get("golden-event-mentions", []):
            event_type = event.get("event_type")
            trigger = event.get("trigger", {}).get("text")

            found_roles = {arg["role"]: arg["text"] for arg in event.get("arguments", [])}
            expected_roles = event_schema.get(event_type, {}).get("roles", [])

            role_details = []
            for role in expected_roles:
                role_details.append({"role": role, "text": found_roles.get(role, "None")})

            event_map[event_type].append({
                "sentence_id": sentence_id,
                "trigger": trigger,
                "arguments": role_details
            })
            sentence_event_map[sentence_id].add(event_type)

    return event_map, sentence_event_map

def create_docs(sentences):
    words, spaces, entity_spans, event_spans = [], [], [], []
    event_map = {}
    entity_map = set()
    event_types = set()
    dep_labels_all, dep_docs = [], []

    offset = 0
    for sent in sentences:
        sent_words = sent["words"]
        words.extend(sent_words)
        spaces.extend([True] * (len(sent_words) - 1) + [True])

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

        dep_doc = nlp(" ".join(sent_words))
        dep_docs.append(dep_doc)
        dep_labels = sorted(set(token.dep_ for token in dep_doc))
        dep_labels_all.append(dep_labels)

        offset += len(sent_words)

    main_doc = Doc(nlp.vocab, words=words, spaces=spaces)
    ner_doc, event_doc = main_doc.copy(), main_doc.copy()
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
    with open("image_only_event.json", encoding="utf-8") as f:
        image_event_data = json.load(f)

    #Match all images related to the current article
    image_entries = {k: v for k, v in image_event_data.items() if k.startswith(article_id)}

    # Extract unique image filenames 
    image_filenames = [f"images/{k}.jpg" for k in image_entries.keys()]  

    relevant = [d for d in data if d.get("sentence_id", "").startswith(article_id)]
    if not relevant:
        return "Article ID not found", 404

    # ADD THIS LINE TO CREATE PLAIN TEXT SENTENCES
    for sent in relevant:
        sent["sentence"] = " ".join(sent["words"])

    events, sentence_event_map = extract_events(relevant, event_schema)
    ner_doc, event_doc, dep_docs, entity_map, dep_labels_all, event_types = create_docs(relevant)

    ner_html = displacy.render(ner_doc, style="ent", options={"colors": COLORS}, page=True)
    event_html = displacy.render(event_doc, style="ent", options={"colors": COLORS}, page=True)
    dep_html_list = [displacy.render(dep, style="dep", page=False) for dep in dep_docs]
    zipped_deps = list(zip(dep_html_list, dep_labels_all))
    # Convert set to list for rendering
    sentence_event_map = {k: list(v) for k, v in sentence_event_map.items()}

    return render_template("full_view.html",
                           article_id=article_id,
                           ner_html=ner_html,
                           event_html=event_html,
                           dep_html_list=dep_html_list,
                           entity_map=entity_map,
                           zipped_deps=zipped_deps,
                           event_types=event_types,
                           events=events,
                           sentence_event_map=sentence_event_map,
                           event_type_colors=EVENT_TYPE_COLORS,
                           color_map=COLORS,
                           relevant=relevant,
                           event_schema=event_schema,
                           image_filenames=image_filenames,
                           sentences=relevant)  # this was likely missing too


if __name__ == "__main__":
    app.run(debug=True)
