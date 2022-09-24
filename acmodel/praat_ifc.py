# Interface to praat

# Functions to convert computed alignment intervals to praat textgrid file format
#
# Use like this:
#   from praat_ifc import textgrid_file_text
#   print(textgrid_file_text({"slova": [(0,2,"dobrĂ˝"), (2,3,"den")], "segmenty": [(0,1,"d"),(1,2,"Ă˝"),(2,2.5,'de'),(2.5,3,'n')]}))


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


def remove_quotes(txt): # small helper proc for textgrid reader below
    assert txt[0] == txt[-1] == '"'
    return txt[1:-1]

def read_interval_tiers_from_textgrid_file(filename):
    """
    Read all interval tiers from a TextGrid file
    """
    with open(filename, 'r') as fileread:
        lines = [l.strip() for l in fileread.readlines()]
    tiers = {}
    state = 'initial'
    for line in lines:
        line_items = line.split()
        #print(f"{state}  {line_items}")
        match line_items:                    # NOTE: Needs Python 3.10
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
                text = remove_quotes(" ".join(text)) # text may be ['"aaa', 'bbb', 'ccc"']
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


desampify_list =[ # First, regular equivalence with our internal phone set
 ('e', 'e'),  ('a', 'a'),  ('o', 'o'),  ('t', 't'),  ('s', 's'),  ('l', 'l'),  ('n', 'n'),
 ('', '|'),  ('i', 'y'),  ('m', 'm'),  ('v', 'v'),  ('i:', 'ý'),  ('r', 'r'),  ('k', 'k'),
 ('p', 'p'),  ('d', 'd'),  ('j', 'j'),  ('u', 'u'), ('a:', 'á'),  ('?', '?'),  ('J', 'ň'),
 ('z', 'z'),  ('b', 'b'),  ('t_s', 'c'),  ('h\\', 'h'), ('S', 'š'),  ('e:', 'é'),  ('x', 'H'),
 ('c', 'ť'),  ('t_S', 'č'),  ('Z', 'ž'),  ('o_u', 'O'),  ('f', 'f'),  ('P\\', 'ř'),  ('u:', 'ú'),
 ('J\\', 'ď'),  ('Q\\', 'Ř'),  ('g', 'g'), ('N', 'N'),  ('o:', 'ó'),  ('a_u', 'A'),  ('G', 'G'),
 ('d_z', 'Z'), ('d_Z', 'Ž'),  
 ('e_u', 'E'), # e_u has strangely low frequency\

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










# To be used in Jupyter notebook:
"""
from IPython.core.magic import register_line_cell_magic
@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
"""

