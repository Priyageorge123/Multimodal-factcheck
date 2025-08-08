# app.py
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
    "NORP": "#c49c94","VALUE": "#c5b0d5"
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
            etype = event["event_type"]
            trig  = event["trigger"]["text"]
            found = {a["role"]: a["text"] for a in event.get("arguments", [])}
            expected = event_schema.get(etype, {}).get("roles", [])
            args = [{"role": r, "text": found.get(r, "None")} for r in expected]

            event_map[etype].append({
                "sentence_id": sentence_id,
                "trigger": trig,
                "arguments": args
            })
            sentence_event_map[sentence_id].add(etype)

    return event_map, sentence_event_map

def create_docs(sentences):
    words, spaces, entity_spans, event_spans = [], [], [], []
    entity_map = set()
    dep_labels_all, dep_docs = [], []
    offset = 0

    for sent in sentences:
        w = sent["words"]
        words.extend(w)
        spaces.extend([True]*(len(w)-1) + [True])

        for ent in sent.get("golden-entity-mentions", []):
            start, end = offset+ent["start"], offset+ent["end"]
            entity_spans.append((start, end, ent["entity-type"]))
            entity_map.add(ent["entity-type"])

        for ev in sent.get("golden-event-mentions", []):
            if ev.get("trigger"):
                start = offset + ev["trigger"]["start"]
                end   = offset + ev["trigger"]["end"]
                event_spans.append((start, end, ev["event_type"]))

        doc = nlp(" ".join(w))
        dep_docs.append(doc)
        dep_labels_all.append(sorted({t.dep_ for t in doc}))

        offset += len(w)

    main = Doc(nlp.vocab, words=words, spaces=spaces)
    ner_doc   = main.copy()
    event_doc = main.copy()
    ner_doc.ents   = [Span(main, a,b, label=l) for a,b,l in entity_spans]
    event_doc.ents = [Span(main, a,b, label=l) for a,b,l in event_spans]

    return ner_doc, event_doc, dep_docs, entity_map, dep_labels_all, set()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400
        if f and allowed_file(f.filename):
            name = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            f.save(path)
            # strip off .txt or .rsd.txt
            article_id = name.rsplit(".", 2)[0]
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
        schema = json.load(f)
    with open("image_only_event.json", encoding="utf-8") as f:
        image_data = json.load(f)

    # filter text
    relevant = [d for d in data if d.get("sentence_id", "").startswith(article_id)]
    if not relevant:
        return "Article ID not found", 404
    for s in relevant:
        s["sentence"] = " ".join(s["words"])

    # extract only those types present in this article
    events, sentence_event_map = extract_events(relevant, schema)
    # but display _all_ possible event types (even if count==0)
    event_types = sorted(schema.keys())


    # build images list
    image_filenames = [f"images/{k}.jpg"
                       for k in image_data if k.startswith(article_id)]

    # create spaCy docs (ignoring its event_types output)
    ner_doc, event_doc, dep_docs, entity_map, dep_labels_all, _ = create_docs(relevant)

    ner_html      = displacy.render(ner_doc,   style="ent",
                                   options={"colors": COLORS}, page=True)
    dep_html_list = [displacy.render(d, style="dep", page=False)
                     for d in dep_docs]
    zipped_deps = list(zip(dep_html_list, dep_labels_all))

    # map each event type → list of sentence texts
    event_sentence_map = defaultdict(list)
    text_map = {s["sentence_id"].split("_")[-1]: s["sentence"]
                for s in relevant}
    for et, evs in events.items():
        for ev in evs:
            sid = ev["sentence_id"]
            if sid in text_map:
                event_sentence_map[et].append({"text": text_map[sid]})

    return render_template("full_view.html",
        article_id=article_id,
        ner_html=ner_html,
        dep_html_list=dep_html_list,
        entity_map=entity_map,
        zipped_deps=zipped_deps,
        events=events,
        sentence_event_map=event_sentence_map,
        event_types=event_types,                # ← your full list here
        event_type_colors=EVENT_TYPE_COLORS,
        color_map=COLORS,
        image_filenames=image_filenames
    )

if __name__ == "__main__":
    app.run(debug=True)
