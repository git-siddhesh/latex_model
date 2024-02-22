import sys
import pandas as pd
from datasets import Dataset 
from transformers import AutoTokenizer
import os
import time
import pickle
# multiprocessing using joblib
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

num_cores = multiprocessing.cpu_count()
print(num_cores)


tokenizer_path = "/mnt/NFS/patidarritesh/SIDDHESH_DATA/hf_tokenizer_1.0%_30000_new" # sentencepiece tokenizer path
data_path="/mnt/NFS/patidarritesh/SIDDHESH_DATA"
block_size = 2048

test_df = pd.read_csv(os.path.join(data_path, 'test.csv'), usecols=['tex','year','month'])
test_df['tex'] = test_df['tex'].apply(lambda x: os.path.join(data_path, "/".join(x.split('/')[6:])))

train_df = pd.read_csv(os.path.join(data_path, 'train.csv'), usecols=['tex','year','month'])
train_df['tex'] = train_df['tex'].apply(lambda x: os.path.join(data_path, "/".join(x.split('/')[6:])))

print(f"Total Test dataset size : ", len(test_df))
print(f"Train dataset size : ", len(train_df))

train_dataset = None
val_dataset = None
test_dataset = None


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '<pad>',
                                    'cls_token': '<cls>',
                                    'sep_token': '<sep>',
                                    'mask_token': '<mask>',
                                    'unk_token': '<unk>',
                                    'bos_token': '<bos>',
                                    'eos_token': '<eos>'
                                })

def group_texts( examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Determine the total length after concatenation
    total_length = len(concatenated_examples[list(examples.keys())[0]])     # len(concatenated_examples['input_ids'])
    num_chunks = total_length // block_size
    # Calculate the remainder
    remainder = total_length % block_size
    
    # Add padding if there is a remainder and the remainder is less than half the block size #IMPORTANT
    if remainder != 0 and 2 * remainder>= block_size:
        num_chunks += 1
        padding_length = block_size - remainder
        for k in concatenated_examples.keys():
            concatenated_examples[k] += [tokenizer.pad_token_id] * padding_length
    # now the length of the concatenated_examples is a multiple of block_size
            

    # # add the num_chunks to the concatenated_examples
    #  = num_chunks

    # Split by chunks of block_size.
    result = { 
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result['num_chunks'] = [num_chunks] + [-1]*(len(result['input_ids'])-1)
    
    return result


def tokenize_function(examples):
    return tokenizer(examples["text"])#.to('cuda:0')

def save_pickel(train_dataset, val_dataset, test_dataset, year, month):
    path = "/mnt/NFS/patidarritesh/SIDDHESH_DATA/data_pk/"
    file_name = f"{year}_{month}_datasets.pkl"
    pickle.dump(train_dataset, open(f"{path}train_{file_name}", "wb"))
    print("Train pickle file saved for year: ", year, " and month: ", month)
    pickle.dump(val_dataset, open(f"{path}val_{file_name}", "wb"))
    print("Val pickle file saved for year: ", year, " and month: ", month)
    pickle.dump(test_dataset, open(f"{path}test_{file_name}", "wb"))
    print("Test pickle file saved for year: ", year, " and month: ", month)
    print("-------------------------------------")

# debug mode will not load the bbl file
def create_dataset_from_files( logger, file_names, month, year, bbl_file_name, bbl_step = 1, isDebug=False):
    '''
    isDebug: if True, then the bbl file will not be loaded, default is False
    bbl_step: the step size to load the bbl file, default is 1, which means all the bbl files will be loaded
    '''
    latex_corpus = []
    print("Reading LaTeX files")
    for file_path in file_names:
        try:
            with open(file_path, 'r', encoding='utf-8') as latex_file:
                latex_corpus.append(latex_file.read())
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as latex_file:
                latex_corpus.append(latex_file.read())

    if not latex_corpus:
        logger.error(f"No LaTeX files found in the Month and Year: {month} {year}")
        print("No latex files found")


    st = time.time()
    logger.info(f"Encoding tokens started...")
    # created a dataframe from a list of latex data
    df_tex = pd.DataFrame(latex_corpus, columns=['text'])
    if isDebug == False:
        df_bbl = pd.read_csv(bbl_file_name , usecols=['bbl'])
        df_bbl.rename(columns={'bbl': 'text'}, inplace=True)
        df = pd.concat([df_tex, df_bbl], ignore_index=True)
        del df_bbl
    else:
        df = df_tex

    del df_tex

    dataset = Dataset.from_pandas(df)

    del latex_corpus

    st = time.time()
    tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True,
                                batch_size=4,
                                remove_columns=['text',
                                                # '__index_level_0__'
                                                ],
                                num_proc=48,
                            )
    print("tokenized_dataset created")

    # convert the tokenized dataset to a list of lists
    latex_corpus_tokenized = tokenized_dataset['input_ids']
    logger.info(f"PAPERS: {len(latex_corpus_tokenized)}, TOTAL TOKENS: {sum([len(x) for x in latex_corpus_tokenized])}")
    logger.info(f"Size of latex_corpus <tokenized>   {sys.getsizeof(latex_corpus_tokenized)}")
    del latex_corpus_tokenized

    et = time.time()
    msg = f"tokenized_dataset created and took {(et - st)/60 :.2f} minutes"
    print(msg)
    print(tokenized_dataset)
    logger.info(msg)
    logger.info(tokenized_dataset)


    lm_datasets = tokenized_dataset.map(
                                    group_texts,
                                    batch_size=1,
                                    batched=True,
                                    num_proc=48,
                            )
    msg = f"lm_datasets created and took {(time.time() - et)/60 :.2f} minutes"
    print(msg)
    print(lm_datasets)
    logger.info(msg)
    logger.info(lm_datasets)

    # remove the 'num_chunks' column from the dataset
    num_chunks = lm_datasets['num_chunks']
    # remove all the values of -1 from the list
    # chunks = []
    total_num_chunks = 0
    for chunk in num_chunks:
        if chunk != -1:
            total_num_chunks += chunk
            # chunks.append(chunk)

    print("Total number of chunks: ", total_num_chunks)
    logger.info(f"Total number of chunks: {total_num_chunks}")

    lm_datasets.remove_columns('num_chunks')
    
    # pickle.dump(lm_datasets, open(f"{dest_path}/{yr}_{month}_lm_datasets.pkl", "wb"))
    # print("lm_datasets dumped to pickle file")
    # logger.info("lm_datasets dumped to pickle file")

    return lm_datasets
    

def generate_dataset_from_files( month, year, logger=None, test_percent=0.5, train_percent=1, val_percent=0.5, isDebug=False):
    '''
    test_percent: the percentage of the test files to be used, default is 0
    train_percent: the percentage of the train files to be used, default is 1
    val_percent: the percentage of the validation files to be used, default is 1
    isDebug: if True, then the bbl file will not be loaded, default is False
    '''
    yymm = int(f'{year}{month}' if month>9 else f'{year}0{month}')
    bbl_yymm = f"0{year}{f'0{month}' if month<10 else f'{month}'}" if year<10 else f"{year}{f'0{month}' if month<10 else f'{month}'}"
    
    train_dataset, val_dataset, test_dataset = None, None, None

    test_total_file_names = test_df[(test_df['year']==year) & (test_df['month']==yymm)]['tex'].to_list()
    bbl_test_file_name = os.path.join(data_path, 'TEST_BBL',f'bbl_split_test_{bbl_yymm}.csv')

    val_file_names = test_total_file_names[:int(len(test_total_file_names)*val_percent)]
    test_file_names = test_total_file_names[int(len(test_total_file_names)*val_percent):int(len(test_total_file_names)*val_percent)+int(len(test_total_file_names)*test_percent)]
    
    
    file_names = train_df[(train_df['year']==year) & (train_df['month']==yymm)]['tex'].to_list()
    train_file_names = file_names[:int(len(file_names)*train_percent)]

    print("-------------------------------------")
    print(f"Total Train file names: for month:{month}, year:{year}:=", len(file_names))
    print(f"Final Train file names: for month:{month}, year:{year}:=", len(train_file_names))
    print("-------------------------------------")
    print(f"Total test file names: for month:{month}, year:{year}:=", len(test_total_file_names))
    print(f"Val file names: for month:{month}, year:{year}:=", len(val_file_names))
    print(f"Test file names: for month:{month}, year:{year}:= ", len(test_file_names))
    print("-------------------------------------")
    logger.info(f"Total Train file names: for month:{month}, year:{year}:= {len(file_names)}")
    logger.info(f"Final Train file names: for month:{month}, year:{year}:= {len(train_file_names)}")
    logger.info(f"Total test file names: for month:{month}, year:{year} :={len(test_total_file_names)}")
    logger.info(f"Val file names: for month:{month}, year:{year}:= {len(val_file_names)}")
    logger.info(f"Test file names: for month:{month}, year:{year}:= {len(test_file_names)}")


    if len(val_file_names) == 0:
        print("No validation files create")
    else:
        val_dataset = create_dataset_from_files(logger, val_file_names, month, year, bbl_test_file_name, bbl_step = 1, isDebug=isDebug)    
    
    if len(test_file_names) == 0:
        print("No test files create")
    else:
        test_dataset = create_dataset_from_files(logger, test_file_names, month, year, bbl_test_file_name, bbl_step = 1, isDebug=isDebug)    
    
    

        


    bbl_file_name = os.path.join(data_path, 'TRAIN_BBL',f'bbl_split_train_{bbl_yymm}.csv')
    
    

    if len(train_file_names) == 0:
        print("No train files create")
    else:
        print(f"Train dataset size: {len(train_file_names)} for year: {year} and month: {month}")
        logger.info(f"Train dataset size: {len(train_file_names)} for year: {year} and month: {month}")
        train_dataset = create_dataset_from_files(logger, train_file_names, month, year, bbl_file_name, bbl_step = 1, isDebug=isDebug)

    save_pickel(train_dataset, val_dataset, test_dataset, year, month)

# create logger
import logging
from datetime import datetime

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
timestamp = "--"
logging.basicConfig(level=logging.NOTSET, filename="{}/Tokenize_log_{}.log".format(data_path, timestamp), filemode="w", format="%(asctime)-15s %(name)-10s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)
# #------------------------------------------------------------------------------------------------------------------


Parallel(n_jobs=4, verbose=4)(delayed(generate_dataset_from_files)(month, 0, logger=logger, test_percent=0.5, train_percent=1, val_percent=0.5, isDebug=False) for yr in range(24) for month in range(1, 13))
Parallel(n_jobs=4, verbose=4)(delayed(generate_dataset_from_files)(month, 1, logger=logger, test_percent=0.5, train_percent=1, val_percent=0.5, isDebug=False) for yr in range(24) for month in range(1, 13))
Parallel(n_jobs=4, verbose=4)(delayed(generate_dataset_from_files)(month, 2, logger=logger, test_percent=0.5, train_percent=1, val_percent=0.5, isDebug=False) for yr in range(24) for month in range(1, 13))
Parallel(n_jobs=4, verbose=4)(delayed(generate_dataset_from_files)(month, 3, logger=logger, test_percent=0.5, train_percent=1, val_percent=0.5, isDebug=False) for yr in range(24) for month in range(1, 13))


#  rsync -ra -e 'ssh -p 2020' --info=progress2  SID_DATA_PROCESSED dosisiddhesh@10.0.62.205:/home/dosisiddhesh/
# dest : /mnt/NFS/patidarritesh/SIDDHESH_DATA
# source : /home/dosisiddhesh/SID_DATA_PROCESSED/DATA_2 
# rsync -ra -e 'ssh -p 2020' --info=progress2  dosisiddhesh@10.0.62.205:/home/dosisiddhesh/SID_DATA_PROCESSED/DATA_2/method2/ /mnt/NFS/patidarritesh/SIDDHESH_DATA/.
