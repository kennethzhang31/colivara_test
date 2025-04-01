import os
import json
import base64
from pathlib import Path
from pdf2image import convert_from_path
from collections import defaultdict
from colivara_py import ColiVara
from dotenv import load_dotenv
load_dotenv()

qid_to_check = [
    2, 4, 11, 19, 23, 24, 29, 50, 51,
    53, 54, 58, 59, 62, 63, 68, 70,
    73, 74, 75, 76, 78, 79, 80, 81,
    82, 83, 89, 92, 93, 94, 96, 98, 99
]

WORKING_DIR = "/Users/kennethzhang/Downloads/vision_llm_testing_data/aicup_dataset"
DATASET = f"{WORKING_DIR}/dataset/preliminary"
SOURCE = f"{WORKING_DIR}/reference"
RESULTS_LOG_PATH = "colivara_results.jsonl"
INDEX_LOG_PATH = "indexed_files.jsonl"
TMP_DIR = Path("./tmp")
TMP_DIR.mkdir(exist_ok=True)

rag_client = ColiVara(
    base_url="https://api.colivara.com",
    api_key=os.getenv("COLIVARA_API_KEY")
)

# LOADS DATA FROM QUESTIONS AND GROUND TRUTH 
def load_data(questions_path, ground_truth_path):
    with open(questions_path, 'r', encoding='utf-8') as qf, \
         open(ground_truth_path, 'r', encoding='utf-8') as gf:
        questions = {q['qid']: q for q in json.load(qf)['questions']}
        ground_truths = {gt['qid']: gt for gt in json.load(gf)['ground_truths']}
    return questions, ground_truths

# LOAD ALREADY INDEX DATA RECORD
def load_indexed_ids(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return {json.loads(line.strip())["file_id"] for line in f if line.strip()}

# LOAD INDEXED DATA TO AVOID REPEATING
def log_indexed_file(file_id, log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump({"file_id": file_id}, f)
        f.write("\n")

# LOG RESULTS FROM RETRIEVAL
def log_result(qid, query, gt_id, retrieved_ids, is_correct):
    print("Querying with filter:", {
        "on": "document",
        "key": "qid_membership",
        "value": f".{qid}.",
        "lookup": "contains"
    })
    result = {
        "qid": qid,
        "query": query,
        "gt_id": gt_id,
        "retrieved_ids": retrieved_ids,
        "is_correct": is_correct
    }
    with open(RESULTS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

# INDEX DOCUMENTS WITH COLIVARA, CONVERTS PDF TO IMAGE, THEN BASE64
def sync_documents(source_ids, category, source_to_qids):
    indexed_ids = load_indexed_ids(INDEX_LOG_PATH)
    folder_path = os.path.join(SOURCE, category)
    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".pdf") and f.split(".")[0].isdigit() and int(f.split(".")[0]) in source_ids
    ]

    for file in files:
        file_id = file.split(".")[0]
        index_key = f"{category}_{file_id}"
        if index_key in indexed_ids:
            print(f"Skipping already indexed: {file_id}")
            continue

        try:
            images = convert_from_path(os.path.join(folder_path, file))
            qid_membership = ''.join([f".{q}." for q in source_to_qids[index_key]])

            for i, img in enumerate(images):
                buffer = TMP_DIR / f"{file_id}_page_{i+1}.png"
                img.save(buffer, format="PNG")
                with open(buffer, "rb") as img_f:
                    encoded = base64.b64encode(img_f.read()).decode("utf-8")

                # METADATA: "qid_membership: ".{qid_1}. .{qid_2}. ,..., .{qid_i}."
                #           to allow filter based on metadata string.
                #           in search, we will see if the value is a substring of the document's key, i.e. qid_membership
                rag_client.upsert_document(
                    name=f"{file_id}_page_{i+1}",
                    document_base64=encoded,
                    collection_name=f"data_{category}_collection",
                    metadata={"qid_membership": qid_membership},
                    wait=True
                )
                print(f"Upserted {category}/{file_id}.pdf")
                buffer.unlink()

            log_indexed_file(index_key, INDEX_LOG_PATH)
        except Exception as e:
            print(f"Failed to process {file_id}: {e}")

# RETRIEVAL LOGIC
def rag_retrieval(query, category, qid):
    return rag_client.search(
        query=query,
        collection_name=f"data_{category}_collection",
        top_k=1,
        query_filter={
            "on": "document",
            "key": "qid_membership",
            "value": f".{qid}.",
            "lookup": "contains"
        }
    ).results

def eval_():
    source_to_qids = defaultdict(set)
    questions, ground_truths = load_data(
        f"{DATASET}/questions.json",
        f"{DATASET}/ground_truths.json"
    )

    for qid, q in questions.items():
        for sid in q["source"]:
            source_to_qids[f"{q['category']}_{sid}"].add(qid)

    for qid in qid_to_check:
        q = questions[qid]
        gt_id = ground_truths[qid]["retrieve"]
        category = ground_truths[qid]['category']

        print(f"Q{qid}: {q['query']}")
        print(f"Sources: {q['source']}")
        print(f"GT: {gt_id}")

        sync_documents(q['source'], category, source_to_qids)
        results = rag_retrieval(q['query'], category, qid)

        top_doc_names = [r.document_name for r in results]
        top_doc_ids = [int(name.split('_')[0]) for name in top_doc_names]
        is_correct = gt_id in top_doc_ids

        print(f"GT: {gt_id} / TOP-K: {top_doc_ids}")
        log_result(qid, q["query"], gt_id, top_doc_ids, is_correct)

if __name__ == '__main__':
    eval()