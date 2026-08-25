import json
import pandas as pd


def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')

    return text


def qm_process_data(train=True):
    ret = []
    data = []
    with open("./data/QMSum/data/ALL/jsonl/{}.jsonl".format("train" if train else "test"), 'r') as file:
        for line in file:
            data.append(json.loads(line))

    for sample in data:
        ret_sample = dict()
        ret_sample["topic_list"] = sample['topic_list']
        ret_sample["general_query_list"] = sample['general_query_list']
        all_transcripts = "\n".join(
            ["Speaker: " + i["speaker"] + "\n" + "Content: " + i["content"] for i in sample['meeting_transcripts']])
        ret_sample["meeting_transcripts"] = clean_data(all_transcripts)
        ret.append(ret_sample)

    return ret


def booksum_process_data(train=True):
    ret = []
    if train:
        data = pd.read_csv('./data/Booksum/train.csv')
    else:
        data = pd.read_csv('./data/Booksum/test.csv')

    for index, row in data.iterrows():
        # Filter out too short text for long-context summarization
        if row["chapter_length"] <= 8000:
            continue
        ret.append(row)

    return ret


def wcep_process_data(train=True):
    ret = []
    data = []
    with open('./data/WCEP/{}.txt'.format("train" if train else "test"), 'r') as file:
        lines = file.readlines()
        for line in lines:
            data.append(json.loads(line))

    for row in data:
        word_num = len(" ".join(row["document"]).split())
        # Filter out too short text for long-context summarization
        if word_num <= 6000:
            continue
        ret.append(row)

    return ret


def gov_process_data(train=True):
    ret = []
    if train:
        data1 = pd.read_parquet('./data/GovReport/document/train-00000-of-00002.parquet')
        data2 = pd.read_parquet('./data/GovReport/document/train-00001-of-00002.parquet')
        data = pd.concat([data1, data2])
    else:
        data = pd.read_parquet('./data/GovReport/document/test-00000-of-00001.parquet')

    for index, row in data.iterrows():
        word_num = len(row["report"].split())
        # Filter out too short text for long-context summarization
        if word_num <= 8000:
            continue
        ret.append(row)

    return ret


def squ_process_data(train=True):
    ret = []
    data = []
    if train:
        # Expand the training set
        with open("./data/SQuALITY/data/v1-3/txt/train.jsonl", 'r') as file:
            for line in file:
                data.append(json.loads(line))
        with open("./data/SQuALITY/data/v1-3/txt/dev.jsonl", 'r') as file:
            for line in file:
                data.append(json.loads(line))
        cnt = 0
        with open("./data/SQuALITY/data/v1-3/txt/test.jsonl", 'r') as file:
            for line in file:
                data.append(json.loads(line))
                cnt += 1
                if cnt == 25:
                    break
    else:
        # Ensure that the test set does not overlap with the training set
        cnt = 0
        with open("./data/SQuALITY/data/v1-3/txt/test.jsonl", 'r') as file:
            for line in file:
                if cnt < 25:
                    cnt += 1
                    continue
                data.append(json.loads(line))

    for sample in data:
        ret.append(sample)

    return ret


def get_processed_data(dataset, train=True):
    if dataset == "qmsum":
        data = qm_process_data(train=train)
    elif dataset == "wcep":
        data = wcep_process_data(train=train)
    elif dataset == "booksum":
        data = booksum_process_data(train=train)
    elif dataset == "govreport":
        data = gov_process_data(train=train)
    elif dataset == "squality":
        data = squ_process_data(train=train)
    else:
        raise Exception("Dataset Error")

    return data


def split_corpus_by_doc(dataset, sample, text_splitter):
    chunk_list = []
    if dataset == "qmsum":
        doc_list = [sample["meeting_transcripts"]]
    elif dataset == "wcep":
        doc_list = sample["document"]
    elif dataset == "booksum":
        doc_list = [sample["chapter"]]
    elif dataset == "govreport":
        doc_list = [sample["report"]]
    elif dataset == "squality":
        doc_list = [sample["document"]]
    else:
        raise Exception("Dataset Error")

    for doc in doc_list:
        chunk_list.extend(text_splitter.split_text(doc))

    return chunk_list


def eval_data_generation(dataset, sample):
    ret = []
    if dataset == "qmsum":
        all_topic = ", ".join([i["topic"] for i in sample["topic_list"]])
        for test_query in sample["general_query_list"]:
            data = dict()
            data["rag_query"] = test_query["query"] + " The topic list of the meeting transcript is: " + all_topic
            data["query"] = test_query["query"]
            data["summary"] = test_query["answer"]
            ret.append(data)
    elif dataset == "wcep":
        data = dict()
        data["rag_query"] = "Summarize the contents of this news event."
        data["query"] = "Summarize the contents of this news event."
        data["summary"] = sample["summary"]
        ret.append(data)
    elif dataset == "booksum":
        data = dict()
        data["rag_query"] = "Summarize the contents of this story."
        data["query"] = "Summarize the contents of this story."
        data["summary"] = sample["summary_text"]
        ret.append(data)
    elif dataset == "govreport":
        data = dict()
        data["rag_query"] = "Summarize the contents of this report."
        data["query"] = "Summarize the contents of this report."
        data["summary"] = sample["summary"]
        ret.append(data)
    elif dataset == "squality":
        data = dict()
        data["rag_query"] = sample["questions"][0]["question_text"]
        data["query"] = sample["questions"][0]["question_text"]
        data["summary"] = [i["response_text"] for i in sample["questions"][0]["responses"]]
        ret.append(data)
    else:
        raise Exception("Dataset Error")

    return ret


if __name__ == '__main__':
    pass
