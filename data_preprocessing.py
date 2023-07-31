import pandas as pd


# file directory:
# |-- eval_retrieval.py
# |-- data_preprocessing.py
# |-- data/ -- eval.pkl
# |        |-- validation_data.tsv
# |-- db/

def preprocessing(df, num_passage=1000):

    df = df.sort_values('relevancy', ascending=False)

    def limit_num_passage(group):

        if sum(group['relevancy']) != 1 or len(group) != 1000:
            return None

        return group[:num_passage]

    result = df.groupby('qid').apply(limit_num_passage).reset_index(drop=True)

    return result


if __name__ == '__main__':

    df = pd.read_csv('data/validation_data.tsv', sep='\t')
    df_filtered = preprocessing(df)
    df_filtered.to_pickle('data/eval.pkl')
