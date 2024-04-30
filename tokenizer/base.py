import unicodedata


def get_stats(ids,count = None):
    '''
    Example = [1,2,3,1,2]  -> {(1,2) -> 2, (2,3)-> 1, (3,1) -> 1 }
    '''
    counts = {} if count is None else count
    
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0) + 1
    
    return counts


def merge(ids,pair, idx):
    '''
    In the list of integers(ids), replace all consecutive occurrence 
    of pair with a new token idx
    Example: id = [1,2,3,1,2]
    so let the pair be (1,2), replacing it with new id as 4 which is idx.
    therefore, new id - [4,3,4]
    '''
    newids = []
    i = 0
    
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids)-1  and ids[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    
    return newids

def replace_control_characters(s:str)-> str:
    '''
    we don't want to print control characters
    which distort the output (e.g. \n or much worse)
    '''
    
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes)->str:
    # pretty print a token, escaping control characters
    s = t.encode('utf-8',errors = 'replace')
    s = replace_control_characters(s)
    return s


# ---------------------------------------------------------------------------
# The base Tokenizer

class Tokenizer:
    