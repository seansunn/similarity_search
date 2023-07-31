import argparse
import os
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


# file directory:
# |-- eval_retrieval.py
# |-- data_preprocessing.py
# |-- data/ -- eval.pkl
# |        |-- validation_data.tsv
# |-- db/

def main():

    # load data
    df = pd.read_pickle(args.data_path)

    # initialize embedding function
    embedding = OpenAIEmbeddings(
        model='text-embedding-ada-002',
    )

    # select first 100 queries for testing
    qids = set(df['qid'].unique()[:args.n_queries])

    # initialize database and get list of stored qids
    db = Chroma(collection_name='eval_100',
            persist_directory=args.db_dir,
            embedding_function=embedding)
    metadata_dict = db.get(include=['metadatas'])['metadatas']
    stored_qid_set = set([d['qid'] for d in metadata_dict])
    not_stored_qids = qids - stored_qid_set

    print(f"Total unstored queries: {len(not_stored_qids)}\n")

    item = 0

    for q in not_stored_qids:

        start = timer()

        df_target = df[df['qid'] == q]

        passage_ids = ['qid_' + str(q) + '_pid_' + x
                       for x in df_target['pid'].astype(str).to_list()]

        passage_list = df_target['passage'].to_list()

        metadatas = [{"qid": str(q)} for _ in range(df_target.shape[0])]

        print(f'Embedding passages for query {item+1}...')
        db.add_texts(
            texts=passage_list,
            metadatas=metadatas,
            ids=passage_ids
        )

        # save database to disk
        db.persist()

        end = timer()
        print(f"Done. Time elapsed: {end - start}\n")

    finish = timer()
    print(f"Job finished. Total time elapsed: {start - finish}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_dir', type=str, default='db')
    parser.add_argument('--data_path', type=str, default='data/eval.pkl')
    parser.add_argument('--api_key', type=str, default='EMPTY')
    parser.add_argument('--n_queries', type=int, default=100)
    args = parser.parse_args()

    # set environment variables
    if args.api_key == 'EMPTY':
        os.environ["OPENAI_API_BASE"] = 'http://localhost:8001/v1'
        os.environ["OPENAI_API_KEY"] = args.api_key
        print('Using FastChat API')
    else:
        os.environ["OPENAI_API_KEY"] = args.api_key
        print('Using OpenAI API')

    main()
