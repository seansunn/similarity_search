from datasets import Dataset, DatasetDict, load_dataset
import re


corpus = load_dataset('BeIR/nfcorpus', 'corpus')
queries = load_dataset('BeIR/nfcorpus', 'queries')
qrels = load_dataset('BeIR/nfcorpus-qrels')


def formatting_docs(sample):
    doc_id = sample['_id']
    sample.pop('_id')
    doc_str = sample['title'] + '. ' + sample['text']
    # doc_str = re.sub(r'\n\s+', ' ', doc_str)
    return {'doc_id': doc_id, 'doc_str': doc_str}


clean_docs = corpus['corpus'].map(formatting_docs).select_columns(['doc_id', 'doc_str'])

qrels_dict = {}
for sample in qrels['test']:
    query_id = sample['query-id']
    doc_id = sample['corpus-id']
    qrels_dict[query_id] = qrels_dict.get(query_id, []) + [doc_id]


qid_list = queries['queries']['_id']
query_id = list(qrels_dict.keys())
query = [queries['queries'][qid_list.index(id)]['text'] for id in query_id]
doc_ids = list(qrels_dict.values())

clean_qrels = Dataset.from_dict({
    'query_id': query_id,
    'query': query,
    'rel_doc_ids': doc_ids
})

merged_dataset = DatasetDict({
    "docs": clean_docs,
    "qrels": clean_qrels,
})

merged_dataset.save_to_disk("data/eval_nf")
