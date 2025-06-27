from flask import Flask, render_template, request
import json
import spacy
from spacy.tokens import Doc, Span
from spacy import displacy
from collections import defaultdict

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")  


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


def extract_events(sentences):
    event_map = defaultdict(list)

    for sent in sentences:
        sentence_id = sent.get("sentence_id", "")

        for event in sent.get("golden-event-mentions", []):
            event_type = event.get("event_type")
            trigger = event.get("trigger", {}).get("text")
            arguments = [{"role": arg["role"], "text": arg["text"]} for arg in event.get("arguments", [])]

            event_map[event_type].append({
                "sentence_id": sentence_id,
                "trigger": trigger,
                "arguments": arguments
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


@app.route("/")
def view_article():
    article_id = request.args.get("article_id")
    if not article_id:
        return "Please provide an article_id", 400

    with open("text_only_event.json", encoding="utf-8") as f:
        data = json.load(f)

    relevant = [d for d in data if d.get("sentence_id", "").startswith(article_id)]
    if not relevant:
        return "Article ID not found", 404
    
    events = extract_events(relevant)

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
