import re

flatten = lambda items: [i for sublist in items if sublist for i in sublist]

def get_link_CID(line: str, current=None) -> list:
    r = re.compile(r'<a class=.+?CID\-([\d]+)?.*?>')
    m = re.findall(r, line)
    if m:
        if current:
            return list(set(m) - set(current))
        else:
            return m
            
    else:
        return []

def get_CID(article: dict) -> str:
    return str(article.get('Record').get('RecordNumber'))