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
    qids = df['qid'].unique()[:args.n_queries]

    # initialize results dataframe
    results = pd.DataFrame(columns=['qid', 'queries', 'correct_psg',
                                    'similarity_search_top', 'rank'])

    item = 0
    irrelevant_results = 0

    for q in qids:

        start = timer()

        df_target = df[df['qid'] == q]
        passage_list = df_target['passage'].to_list()

        # initialize/load ChromaDB
        db = Chroma(collection_name='eval_100',
                    persist_directory=args.db_dir,
                    embedding_function=embedding)

        print(f'Conducting similarity search for query {item+1}')
        query = df_target['queries'].iloc[0]
        docs = db.similarity_search_with_score(query, k=100, filter=str(q))
        docs_content = [doc.page_content for doc, _ in docs]

        if passage_list[0] in docs_content:
            rank = docs_content.index(passage_list[0]) + 1.
        else:
            rank = np.inf
            print("Couldn't find the relevant passage in top 100 results.")
            irrelevant_results += 1

        item += 1

        end = timer()

        # save results to dataframe and backup
        results.loc[len(results)] = [q, query, passage_list[0],
                                     docs_content[0], rank]
        results.to_csv('results.csv', index=False)

        print(f'Query {item} completed; time elapsed: {(end - start):.2f}s\n')

    # since only one relevant passage for each query,
    # calculating MAP as:
    # (1/n) * sum( 1/rank for query 1 to n)
    print(f'Similarity search for {args.n_queries} queries completed.')
    rank_list = results['rank'].to_numpy()
    mean_ap = np.mean(np.reciprocal(rank_list))
    print(f'Search results out of top 100: {irrelevant_results}')
    print(f'Mean average precision: {mean_ap}')


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
