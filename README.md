import numpy as np
from numba import jit, gdb_init
import re
# Define a custom type for the results

import pandas as pd
import numpy as np
import numba
df=pd.read_csv('/kaggle/input/narrative/your_data.csv')
df=df.head(10)
#text=np.array(df['Narrative'],dtype=np.object_)
text=np.array(df['Narrative'], dtype='S1000')

result_dtype = np.dtype([('keyword', 'S50'), ('matches', 'S50', (10,))])

@jit(nopython=False,debug=True)
def find_pattern(text, keywords, pattern):
    text = text.lower()
    results = np.empty(len(keywords), dtype=result_dtype)
    result_idx = 0
    for keyword in keywords:
        start_pos = 0
        while True:
            keyword_pos = text.find(keyword, start_pos)
            if keyword_pos == -1:
                break
            # Start searching after the keyword
            start_search_pos = keyword_pos + len(keyword)
            subsequent_text = text[start_search_pos:start_search_pos + 50]
            matches = re.findall(pattern, subsequent_text)
            if matches:
                results[result_idx]['keyword'] = keyword
                results[result_idx]['matches'][:len(matches[:10])] = matches[:10]
                result_idx += 1
            start_pos = keyword_pos + 1  # Move start_pos ahead to continue searching the text
    return results[:result_idx]

@jit(nopython=False,debug=True)
def find_patterns_after_keywords(data, keywords, pattern):
    gdb_init()
    results = np.empty(len(data), dtype=result_dtype)
    
    for idx in range(len(data)):
        text = str(data[idx])
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        result = find_pattern(text, keywords, pattern)
        results[idx] = result
    return results


pattern = r'\b\w+\s*/\s*\w+\b'


results = find_patterns_after_keywords(text, keywords, pattern)
print(results)
