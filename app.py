import os
import argparse
import requests
import spacy
from flask import Flask, render_template, request, send_from_directory
from bs4 import BeautifulSoup
from google.cloud import vision
from werkzeug.utils import secure_filename
from spacy import displacy


UPLOAD_FOLDER="C:/Users/priya/IML_Task/uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

cse_id = None
api_key = None

def analyze_image(image_url=False,path=False):
    client = vision.ImageAnnotatorClient()
    if image_url:
        response = requests.get(image_url,timeout=60)
        response.raise_for_status()
        image = vision.Image(content=response.content)
    if  path:
        with open(path,"rb") as image_file:
            content=image_file.read()
        image = vision.Image(content=content)
    response = client.web_detection(image=image)
    annotations = response.web_detection

    results = {"entities": [], "matching_images": [], "visually_similar_images":[],"pages": []}
    if annotations.web_entities:
        results["entities"] = [
            {"description": e.description, "score": e.score}
            for e in annotations.web_entities if e.description
        ]
    if annotations.visually_similar_images:
        results["visually_similar_images"] = [img.url for img in annotations.visually_similar_images]
    if annotations.pages_with_matching_images:
        results["pages"] = [page.url for page in annotations.pages_with_matching_images]
        for page in annotations.pages_with_matching_images:
            if page.full_matching_images:
                results["matching_images"].extend([img.url for img in page.full_matching_images])
            if page.partial_matching_images:
                for image in page.partial_matching_images:
                    results["matching_images"].append(image.url)
    
    return results

def extract_page_text(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n\n".join(p.get_text(strip=True) for p in paragraphs)
    except Exception as e:
        return f"[Error extracting text: {e}]"

def store_image_urls(image_urls, file_path="similar_images.txt"):
    with open(file_path, "a", encoding="utf-8") as file:
        for url in image_urls:
            file.write(url + "\n")

def search_caption(caption):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": caption,
        "cx": cse_id,
        "key": api_key,
        "searchType": "image",
        "num": 5
    }
    response = requests.get(search_url, params=params)
    data = response.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "snippet": item.get("snippet"),
            "page_url":item.get("image").get("contextLink"),
            "link": item.get("link"),
            "image": item.get("image", {}).get("thumbnailLink")
        })
    return results

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],filename)


@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None
    image_file = None
    caption = None
    if request.method == "POST":
        submit_type=request.form.get("submit_type")
        if submit_type=="Search Image":
            image_file=request.files['file']
            image_url = request.form.get("image_url")
        elif submit_type=="Search Caption":
            caption = request.form.get("caption")
        
        filename = None
        path = None
        colors = {
                            "PERSON": "#a6e22d",      
                            "ORG": "#ff7f0e",          
                            "LOC": "#1f77b4",         
                            "FAC": "#d62728",        
                            "PRODUCT": "#9467bd",     
                            "EVENT": "#bcbd22",     
                            "WORK_OF_ART": "#17becf",  
                            "LAW": "#8c564b",        
                            "LANGUAGE": "#e377c2",     
                            "DATE": "#2ca02c",        
                            "TIME": "#1c9099",   
                            "PERCENT": "#7f7f7f",      
                            "MONEY": "#ff1493",       
                            "QUANTITY": "#aec7e8",  
                            "ORDINAL": "#ffbb78",     
                            "CARDINAL": "#98df8a",    
                            "NORP": "#c49c94"         
                        }
        options={"colors":colors}

        if image_url or image_file:
            try:

                if image_file and image_file.filename!="":
                    filename=secure_filename(image_file.filename)
                    path=os.path.join(app.config["UPLOAD_FOLDER"],filename)
                    image_file.save(path)
                    results=analyze_image(path=path)

                else:
                    results = analyze_image(image_url=image_url)

                pages = [{"url":url,"text":extract_page_text(url)} for url in results.get("pages", [])]
                nlp=spacy.load("en_core_web_sm")

                for page in pages:
                    doc = nlp(page["text"])
                    entity_labels = set(ent.label_ for ent in doc.ents)
                    page["colors"] = {label: colors.get(label, "#ddd") for label in entity_labels}
                    # Render HTML for NER and dependency parsing
                    page["ner_html"] = displacy.render(doc, style="ent",options=options, page=True)
                    page["dep_html"] = displacy.render(doc, style="dep", page=True)
                return render_template("result.html",
                                       image_url=image_url,
                                       path=path,
                                       filename=filename,
                                       entities=results["entities"],
                                       matching_images=results["matching_images"],
                                       visually_similar_images=results["visually_similar_images"],
                                       pages=pages
                                       )               
            except Exception as e:
                return f"<h2>Error: {e}</h2>"

        elif caption:
            try:
                search_results = search_caption(caption)
                nlp=spacy.load("en_core_web_sm")
                for result in search_results:
                    result["text"]=extract_page_text(result.get("page_url",""))
                    doc=nlp(result["text"])
                    entity_labels = set(ent.label_ for ent in doc.ents)
                    result["colors"] = {label: colors.get(label, "#ddd") for label in entity_labels}
                    result["rendered_ner"]=displacy.render(doc,style="ent",options=options,page=False)
                    result["rendered_dep"]=displacy.render(doc,style="dep",page=False)
                return render_template("result.html", caption=caption, search_results=search_results)
            except Exception as e:
                return f"<h2>Error: {e}</h2>"

    return render_template("index.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cse_id", required=True, help="Google Custom Search Engine ID")
    parser.add_argument("--api_key", required=True, help="Google API Key")
    args = parser.parse_args()

    cse_id = args.cse_id
    api_key = args.api_key

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/priya/IML_Task/polished-vault-455713-q8-199191d8dfb6.json"

    app.run(debug=True)
