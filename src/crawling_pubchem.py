import sys
sys.path.append('../util/')

import pandas as pd
import requests
import json
import time
from tqdm import tqdm

import pubchem_util as pil

WAIT_TIME = 0.2

def main():
    cid_list, crawling_cid_list = init_CID()
    steps = 2
    for _ in range(steps):
        if not crawling_cid_list:
            break

        # pubchemから記事をクローリング
        internal_links = crawling(crawling_cid_list)

        # 取得した記事のCIDを追加
        cid_list = cid_list + crawling_cid_list

        # 未取得のCIDリストを作成
        crawling_cid_list = get_new_CID(cid_list, internal_links)

    print("Total pages:", len(cid_list))


def crawling(cids):
    internal_link_cids = []
    for cid in tqdm(cids):
        count = 0
        while True:
            if count > 5: break
            try:
                count += 1
                resp = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON')
                time.sleep(WAIT_TIME)
                break
            except requests.ConnectTimeout as e:
                print("Connection timeout.\nRetry...")
                time.sleep(10)
                continue
            except requests.HTTPError as e:
                print("HTTP error.")
                time.sleep(WAIT_TIME)
                break
            except requests.ConnectionError as e:
                print("Connection error.")
                time.sleep(WAIT_TIME)
                break

        if resp is None or resp.status_code == 404:
            continue

        article = resp.json()
        output(article)

        internal_link_cids.append(pil.get_link_CID(article.text))

    internal_link_cids = pil.flatten(internal_link_cids)
    
    return list(set(internal_link_cids))


def output(article):
    filepath = "../data/pubchem_articles.jsonl"
    with open(filepath, 'a') as f:
        json.dump(article, f)
        f.write('\n', f)


def get_new_CID(cids, internal_links):
    return list(set(internal_links) - set(cids))


def init_CID():
    cid_list = []
    link_list = []
    with open("../data/pubchem_articles.jsonl", 'r') as f:
        line = f.readline()
        while line:
            article = json.loads(line.strip())
            cid = pil.get_CID(article)
            cid_list.append(cid)
            link_list.append(pil.get_link_CID(line, cid))
            
            line = f.readline()

    return cid_list, get_new_CID(cid_list, pil.flatten(link_list))

            
if __name__ == '__main__':
    main()