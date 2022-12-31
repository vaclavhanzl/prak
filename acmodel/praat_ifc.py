
# Interface to praat

# Functions to convert computed alignment intervals to praat textgrid file format
#
# Use like this:
#   from praat_ifc import textgrid_file_text
#   print(textgrid_file_text({"slova": [(0,2,"dobrĂ˝"), (2,3,"den")], "segmenty": [(0,1,"d"),(1,2,"Ă˝"),(2,2.5,'de'),(2.5,3,'n')]}))

import sys

from prongen.prak_prongen import clean_textline

def textgrid_file_intro(xmax, size):
    return f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = {xmax} 
tiers? <exists> 
size = {size} 
item []: 
"""

def textgrid_tier_intro(n, name, xmax, num_of_intervals):
    return f"""    item [{n}]:
        class = "IntervalTier" 
        name = "{name}" 
        xmin = 0 
        xmax = {xmax} 
        intervals: size = {num_of_intervals} 
"""

def textgrid_interval(n, xmin, xmax, text):
    return f"""        intervals [{n}]:
            xmin = {xmin} 
            xmax = {xmax} 
            text = "{text}" 
"""

def textgrid_tier(tier_n, name, intervals):
    txt_intervals = ""
    for n, (xmin, xmax, text) in enumerate(intervals, start=1): # we will also use the last xmax and n outside this loop
        txt_intervals += textgrid_interval(n, xmin, xmax, text)
        
    txt = textgrid_tier_intro(tier_n, name, xmax, n) # using maxima from loop for xmax and n
    txt += txt_intervals
    return txt

def textgrid_file_text(tiers):
    """
    Convert time intervals with words or phones to  praat utf8 textgrid format.
    Each key in tiers dictionary becomes a tier name. Corresponding value should
    be a list of intervals in format (from,to,text).
    """
    txt_tiers = ""
    for tier_n, (tier_name, tier_intervals) in enumerate(tiers.items(), start=1): # tier_n and tier_intervals also used after this loop
        #print(f"{tier_n, (tier_name, tier_intervals)=}")
        txt_tiers += textgrid_tier(tier_n, tier_name, tier_intervals)
    _, xmax, _ = tier_intervals[-1] # use end of last interval of last tier - SHOULD USE MAX OF ALL TIERS?
    txt = textgrid_file_intro(xmax, tier_n)
    txt += txt_tiers
    return txt


def count_trailing_doublequotes(txt):
    """
    Count trailing '"' characters in a string. If this number is odd,
    the text ends by a proper termination of a praat string. (Two
    quotes are quoted quote in praat.)
    """
    count = 0
    for i in range(len(txt)-1,-1,-1):
        if txt[i]!='"':
            break
        count += 1
    return count

def praat_string_terminated(txt_list):
    """
    True if txt_list ends by a proper termination of a praat string. (Two
    quotes are quoted quote in praat and this would NOT be a termination,
    unless the interval text is empty, written as two quotes.)
    """
    return txt_list==['""'] or count_trailing_doublequotes(txt_list[-1])%2==1

def remove_quotes(txt): # small helper proc for textgrid reader below
    assert txt[0] == txt[-1] == '"'
    return txt[1:-1]

def read_interval_tiers_from_textgrid_file(filename):
    """
    Read all interval tiers from a TextGrid file
    """
    already_warned_about_newlines = False
    with open(filename, 'r', encoding='utf-8') as fileread: # explicit utf-8 needed on Windows
        lines = [l.strip() for l in fileread.readlines()]
    tiers = {}
    state = 'initial'
    for line in lines:
        line_items = line.split()
        #print(f"{state}  {line_items}")
        match line_items:                    # NOTE: Needs Python 3.10
            # this should be the first case
            case [*add_text] if state=='finishing_string':
                text += add_text
                if praat_string_terminated(text):
                    text = remove_quotes(" ".join(text)) # text may be ['"aaa', 'bbb', 'ccc"']
                    text = clean_textline(text)
                    interval = (float(xmin), float(xmax), text)
                    #print(f"Got interval: {interval}")
                    tiers[tiername].append(interval) # We just left any '""' inside, will be deleted
                    state = 'reading_intervals'
                # else, OMG, another newline in interval text! Stay in finishing_string state.
            case ['class', '=', classname]:
                classname = remove_quotes(classname)
                #print(f"===== {classname} =====")
                if classname == "IntervalTier":
                    state = 'awaiting_tiername'
            case ['name', '=', tiername] if state=='awaiting_tiername':
                tiername = remove_quotes(tiername)
                #print(f"Yes!!! IntervalTier {tiername}")         
                tiers[tiername] = []
                state = 'awaiting_intervals'      
            case ['intervals:', 'size', '=', intervals_size] if state=='awaiting_intervals':
                state = 'reading_intervals'
            case ['xmin', '=', xmin] if state=='reading_intervals':
                pass # getting xmin is what we beeded here
            case ['xmax', '=', xmax] if state=='reading_intervals':
                pass # dtto, got xmax
            case ['text', '=', *text] if state=='reading_intervals':
                if not praat_string_terminated(text): # OMG, newline in interval text!
                    # keeping text list for later expansion
                    state = 'finishing_string'
                    if not already_warned_about_newlines:
                        print(f'Prak WARNING: Found newline after "...{text[-1]}" in tier "{tiername}". Replaced by space.', file=sys.stderr)
                        already_warned_about_newlines = True # do nor repeat more than once per file
                else:
                    text = remove_quotes(" ".join(text)) # text may be ['"aaa', 'bbb', 'ccc"']
                    text = clean_textline(text)
                    interval = (float(xmin), float(xmax), text)
                    #print(f"Got interval: {interval}")
                    tiers[tiername].append(interval)

    return tiers


def read_phone_tier_from_textgrid_file(filename):
    """
    Read TextGrid file and return a phone tier, which may have
    name "Phone" or "phone".
    """
    tiers = read_interval_tiers_from_textgrid_file(filename)
    assert "phone" in tiers or "Phone" in tiers or "segmenty" in tiers
    if "phone" in tiers:
        return tiers["phone"]
    if "Phone" in tiers:
        return tiers["Phone"]
    if "segmenty" in tiers:
        return tiers["segmenty"]
    return None

def read_word_tier_from_textgrid_file(filename):
    """
    Read TextGrid file and return a word tier, which may have
    name "Word" or "word".
    """
    tiers = read_interval_tiers_from_textgrid_file(filename)
    assert "word" in tiers or "Word" in tiers or "slova" in tiers
    if "word" in tiers:
        return tiers["word"]
    if "Word" in tiers:
        return tiers["Word"]
    if "slova" in tiers:
        return tiers["slova"]
    return None


desampify_regular =[ # First, regular equivalence with our internal phone set
 ('e', 'e'),  ('a', 'a'),  ('o', 'o'),  ('t', 't'),  ('s', 's'),  ('l', 'l'),  ('n', 'n'),
 ('', '|'),  ('i', 'y'),  ('m', 'm'),  ('v', 'v'),  ('i:', 'ý'),  ('r', 'r'),  ('k', 'k'),
 ('p', 'p'),  ('d', 'd'),  ('j', 'j'),  ('u', 'u'), ('a:', 'á'),  ('?', '?'),  ('J', 'ň'),
 ('z', 'z'),  ('b', 'b'),  ('t_s', 'c'),  ('h\\', 'h'), ('S', 'š'),  ('e:', 'é'),  ('x', 'H'),
 ('c', 'ť'),  ('t_S', 'č'),  ('Z', 'ž'),  ('o_u', 'O'),  ('f', 'f'),  ('P\\', 'ř'),  ('u:', 'ú'),
 ('J\\', 'ď'),  ('Q\\', 'Ř'),  ('g', 'g'), ('N', 'N'),  ('o:', 'ó'),  ('a_u', 'A'),  ('G', 'G'),
 ('d_z', 'Z'), ('d_Z', 'Ž'),  
 ('e_u', 'E')] # e_u has strangely low frequency\

desampify_list = desampify_regular+[
 # guess-fixing errors, where it likely should have been one of our regular phones:
 (' ?', '?'),  ('ʔ', '?'),   ('e ', 'e'),  ('a ', 'a'),  ('S ', 'š'),   (' p', 'p'),  ('š', 'š'),
 ('č', 'č'),  ('é', 'é'),  ('ə', '@'), ('h\\ ', 'h'),  (' x', 'H'),  (' t', 't'),  ('r ', 'r'),
 ('o_u ', 'O'),  ('? ', '?'),  ('n ', 'n'),   ('m ', 'm'),  ('e: ', 'é'),  ('a: ', 'á'),  ('v ', 'v'),
 ('{pause}', '|'), 

 # Things which make sense but do not directly correspond to our phone set:
 ('I', 'y'),  ('E', 'e'), ('r=', 'r'), ('l=', 'l'),  ('E:', 'é'), ('F', 'm'),  (' ', '|'), ('E_u', 'E'),  
 ('X', 'G'), # ? check in data what X really means
 ('@', '@'), ('@:', '@'), # schwa, we should rethink what to do with it (it is NOT in our phone set)
 ('m=', 'm'),  ('m:', 'm'), # these are very rare

 # Strange things with considerable frequency:
 ('\\sw', '#'), ('h', '#'), ('?\\~v', '#'),    ('R0', '#'), 

 # Low frequency strange rest:
 ('e_a', '#'), ('T', '#'),  ('w', '#'),  ('R', '#'), ('D', '#'), ('?/h', '#'),
 ('?@', '#'),  ('\\nj', '#'), ('xxx', '#'),  ('dZ', '#'),  
 ('tS', '#'), ('j ', '#'), ('\\tf', '#'), ('\\soo', '#'),
 ('\\ic', '#'),  ('V', '#'),  ('J:', '#'), ('@m', '#'),  ('?/L', '#') ]

# Full list of our phones (all are used above):
# '? A E G H N O Z a b c d e f g h j k l m n o p r s t u v y z | á é ó ú ý č ď ň Ř ř š ť Ž ž'

sampify_dict = {}
for sampa_phone, phone in desampify_regular:
    sampify_dict[phone] = sampa_phone


desampify_dict = {}
for sampa_phone, phone in desampify_list:
    desampify_dict[sampa_phone] = phone

def desampify_phone_tier(tier, desampify_dict=desampify_dict):
    """
    Convert phones to our internal alphabet.
    The tier is represented as list of (xmin, xmax, phone).
    """
    out_tier = []
    for (xmin, xmax, phone) in tier:
        if phone in desampify_dict:
            phone = desampify_dict[phone]
        else:
            phone = '#' # general marker for strange things
        out_tier.append((xmin, xmax, phone))
    return out_tier



def sampify_tier(tier):
    """
    Convert phone tier from our internal phone alphabet to SAMPA
    """
    return desampify_phone_tier(tier, desampify_dict=sampify_dict)



def unify_tier_ends(tg_dict, leading_tier, max_fuzz=0.1):
    """
    Slightly move ends of last intervals of all tiers to match the leading tier.
    """
    assert len(leading_tier)>=1 # at least one interval needed
    (_, end, _) = leading_tier[-1] # tier is a list of (xmin, xmax, phone)

    for tier_name, tier_list in tg_dict.items(): # includes leading tier but no problem
        assert len(tier_list)>=1 # at least one interval needed
        (xmin, xmax, text) = tier_list[-1] # take last interval
        if abs(xmax-end)<=max_fuzz and xmin<end: # not much fuzz, and positive interval will remain
            tier_list[-1] = (xmin, end, text) # put it back modified
        else:
            print(f'Prak WARNING: Failed to unify exact end of tier "{tier_name}" with leading tier".', file=sys.stderr)
            if not abs(xmax-end)<=max_fuzz:
                print(f' Ends differ more than {max_fuzz}s.', file=sys.stderr)
            if not xmin<end:
                print(f' Last interval would be negative.', file=sys.stderr)





def split_by_char_unless_quoted(text, split_char=':', escape_char='\\'):
    """
    Split text to list of strings at occurences of split char ":".
    Escape char (backslash) quotes (only) the split char and itself.
    Misplaced escape chars are removed and warning is printed.
    """
    retval = []
    after_e_ch = False
    chunk = ""
    esc_error = False
    for ch in text:
        if after_e_ch:
            chunk += ch
            after_e_ch = False
            if ch!=escape_char and ch!=split_char: # nothing else can be escaped
                esc_error = True
            continue
        if ch==escape_char:
            after_e_ch = True
            continue
        if ch==split_char:
            retval.append(chunk)
            chunk = ""
            continue
        chunk += ch
    if after_e_ch:
        esc_error = True # trailing escape with no char after it
    retval.append(chunk)
    if esc_error:
        print(f"Detected stray escape character '{escape_char}' in input '{text}'.")
        print(f"Prak WARNING: Split the above text as '{(' '+split_char+' ').join(retval)}' but maybe it is not what you wanted.")
    return retval




def rename_tier_by_spec(tier_name, rename_spec_list):
    """
    Compute new name for a tier according to rename_spec_list which is
    iterated till the first match and renamed acording to it.
    If there is no match, None is returned (meaning to discard the tier).
    Rules can have the following form:
       'name'    .. just keep this name
       'old:new' .. replace old name by new
       ':suffix' .. rename anything by adding a suffix
       ':'       .. keep anything as is (special case of the suffix ruile)
    There should be at most one separating ':' in the rule. Tiers containing
    a ':' must use backslash to escape this non-separating ':'.
    """
    for spec in rename_spec_list:
        spec = split_by_char_unless_quoted(spec) # split on ':'
        if len(spec)==1 and tier_name==spec[0]: # 'name'
            return tier_name
        if len(spec)==2 and len(spec[0])==0: # ':suffix'
            return tier_name+spec[1]
        if len(spec)==2 and tier_name==spec[0]: # 'old:new'
            return spec[1]
    return None



def rename_prune_tiers(tiers, rename_spec_list):
    """
    Rename tiers in dictionary-represented TextGrid according to rename
    specification, which is a list of rules applied to each key.
    Tiers with no match in rename_spec_list are pruned away.
    """
    return {tnm2: tls for tnm, tls in tiers.items() if (tnm2:=rename_tier_by_spec(tnm, rename_spec_list))!=None}








# To be used in Jupyter notebook:
"""
from IPython.core.magic import register_line_cell_magic
@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
"""

