from flask import Flask, render_template, request, redirect, url_for
import os, re, json
from werkzeug.utils import secure_filename
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

# ---- Colors ----
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

# ---- Helpers ----
def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_events(sentences, event_schema):
    """Return:
       events: {event_type: [{"sentence_id": "0", "trigger": str, "arguments":[...]}]}
       sentence_event_map: {sid: set(event_types)}  (not used in UI but kept)
    """
    events = defaultdict(list)
    sentence_event_map = defaultdict(set)
    for sent in sentences:
        full_id = sent.get("sentence_id", "")
        sid = full_id.split("_")[-1] if full_id else ""
        for ev in sent.get("golden-event-mentions", []):
            etype = ev.get("event_type")
            trig = (ev.get("trigger") or {}).get("text", "")
            found = {a["role"]: a["text"] for a in ev.get("arguments", [])}
            expected = event_schema.get(etype, {}).get("roles", [])
            args = [{"role": r, "text": found.get(r, "None")} for r in expected]
            events[etype].append({"sentence_id": sid, "trigger": trig, "arguments": args})
            sentence_event_map[sid].add(etype)
    return events, sentence_event_map

def create_docs(sentences):
    """Build displaCy docs; filter overlapping spans"""
    words, spaces = [], []
    raw_entity_spans, raw_event_spans = [], []
    entity_labels = set()
    dep_docs, dep_labels_all = [], []
    offset = 0

    for s in sentences:
        w = s["words"]
        words.extend(w)
        spaces.extend([True] * (len(w) - 1) + [True])

        for ent in s.get("golden-entity-mentions", []):
            start = offset + ent["start"]; end = offset + ent["end"]
            raw_entity_spans.append((start, end, ent["entity-type"]))
            entity_labels.add(ent["entity-type"])

        for ev in s.get("golden-event-mentions", []):
            trig = ev.get("trigger")
            if trig:
                start = offset + trig["start"]; end = offset + trig["end"]
                raw_event_spans.append((start, end, ev["event_type"]))

        # dependency doc for the sentence
        dep_doc = nlp(" ".join(w))
        dep_docs.append(dep_doc)
        dep_labels_all.append(sorted({t.dep_ for t in dep_doc}))

        offset += len(w)

    # filter overlaps (keep longer spans first at same start)
    def filter_spans(raw):
        out, used = [], set()
        for a, b, lab in sorted(raw, key=lambda x: (x[0], -(x[1]-x[0]))):
            tok_ids = set(range(a, b))
            if tok_ids & used:
                continue
            out.append((a, b, lab))
            used |= tok_ids
        return out

    ent_spans = filter_spans(raw_entity_spans)
    evt_spans = filter_spans(raw_event_spans)

    main_doc = Doc(nlp.vocab, words=words, spaces=spaces)
    ner_doc = main_doc.copy()
    ner_doc.ents = [Span(main_doc, a, b, label=lab) for a, b, lab in ent_spans]
    event_doc = main_doc.copy()
    event_doc.ents = [Span(main_doc, a, b, label=lab) for a, b, lab in evt_spans]

    return ner_doc, event_doc, dep_docs, entity_labels, dep_labels_all

# Routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400
        if allowed_file(f.filename):
            name = secure_filename(f.filename)
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], name))
            # Strip .txt then optional .rsd to get article_id 
            base = re.sub(r"\.txt$", "", name, flags=re.I)
            article_id = re.sub(r"\.rsd$", "", base, flags=re.I)
            return redirect(url_for("view_article", article_id=article_id))
    return render_template("upload.html")

@app.route("/view")
def view_article():
    article_id = request.args.get("article_id")
    if not article_id:
        return "Please provide an article_id", 400

    # Load data 
    with open("text_only_event.json", encoding="utf-8") as f:
        text_data = json.load(f)
    with open("event_schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    with open("image_only_event.json", encoding="utf-8") as f:
        img_only = json.load(f)
    try:
        with open("image_multimedia_event.json", encoding="utf-8") as f:
            img_multi = json.load(f)
    except FileNotFoundError:
        img_multi = {}

    # Select sentences for this article 
    relevant = [d for d in text_data if d.get("sentence_id", "").startswith(article_id)]
    if not relevant:
        return "Article ID not found", 404
    for s in relevant:
        s["sentence"] = " ".join(s["words"])

    # Extract text events 
    events, _ = extract_events(relevant, schema)

    # Build event to list of sentence texts in the same order as events[event_type]
    sentence_text_map = {s["sentence_id"].split("_")[-1]: s["sentence"] for s in relevant}
    event_sentence_map = defaultdict(list)
    for et, lst in events.items():
        for ev in lst:
            sid = ev["sentence_id"]
            txt = sentence_text_map.get(sid)
            if txt:
                event_sentence_map[et].append({"text": txt})
    event_sentence_map = dict(event_sentence_map)

    # spaCy visualizations 
    ner_doc, event_doc, dep_docs, entity_labels, dep_labels_all = create_docs(relevant)
    ner_html = displacy.render(ner_doc, style="ent", options={"colors": COLORS}, page=False) 
    dep_html_list = [displacy.render(d, style="dep", page=False) for d in dep_docs]
    zipped_deps = list(zip(dep_html_list, dep_labels_all))

    #  Images: merge from both JSONs and group by event type 
    combined_imgs = {}
    for src in (img_only, img_multi):
        for key, val in src.items():
            if not key.startswith(article_id):
                continue
            if key not in combined_imgs:
                combined_imgs[key] = {"event_type": val.get("event_type"), "role": {}}
            for role, boxes in (val.get("role") or {}).items():
                combined_imgs[key]["role"].setdefault(role, []).extend(boxes)

    # Build objects and group by event type for 3-pane UI
    image_by_type = defaultdict(list)
    for key, val in combined_imgs.items():
        boxes = []
        for role, arrs in (val.get("role") or {}).items():
            for arr in arrs:
                # format: ["1", x1, y1, x2, y2]
                obj_id = str(arr[0]) if len(arr) >= 1 else ""
                x1, y1, x2, y2 = (int(arr[1]), int(arr[2]), int(arr[3]), int(arr[4])) if len(arr) >= 5 else (0, 0, 0, 0)
                boxes.append({"role": role, "obj_id": obj_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        im_obj = {
            "id": key,
            "src": f"images/{key}.jpg",
            "url": url_for("static", filename=f"images/{key}.jpg"),
            "event_type": val.get("event_type"),
            "boxes": boxes
        }
        et = im_obj["event_type"]
        if et:
            image_by_type[et].append(im_obj)
    image_by_type = dict(image_by_type)

    # Event types present (text or images)
    present_event_types = sorted(set(events.keys()) | set(image_by_type.keys()))

    # Render 
    return render_template(
        "full_view.html",
        article_id=article_id,
        ner_html=ner_html,
        zipped_deps=zipped_deps,
        entity_map=entity_labels,
        color_map=COLORS,
        events=events,
        event_sentence_map=event_sentence_map,
        present_event_types=present_event_types,
        event_type_colors=EVENT_TYPE_COLORS,
        image_by_type=image_by_type
    )

if __name__ == "__main__":
    app.run(debug=True)