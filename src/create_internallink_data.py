import sys
sys.path.append('../util/')

import pandas as pd
import json
import pubchem_util as pil

internal_link_df = pd.DataFrame()

with open("../data/pubchem_articles.jsonl", 'r') as f:
    line = f.readline()
    while line:
        article = json.loads(line.strip())
        cid = str(pil.get_CID(article))

        internal_link_df = internal_link_df.append(
            pd.DataFrame({'entry':cid, 'linked': pil.get_link_CID(line, current=cid)})
            , ignore_index=True
        )
        
        line = f.readline()

internal_link_df = internal_link_df[(internal_link_df.entry.str.len() > 0) & (internal_link_df.linked.str.len() > 0)]

internal_link_df.to_csv("../data/internal_CID_link_in_entries.csv", index=False)