# Interface to praat

# Functions to convert computed alignment intervals to praat textgrid file format
#
# Use like this:
#   from praat_ifc import textgrid_file_text
#   print(textgrid_file_text({"slova": [(0,2,"dobrý"), (2,3,"den")], "segmenty": [(0,1,"d"),(1,2,"ý"),(2,2.5,'de'),(2.5,3,'n')]}))


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
        print(f"{tier_n, (tier_name, tier_intervals)=}")
        txt_tiers += textgrid_tier(tier_n, tier_name, tier_intervals)
    _, xmax, _ = tier_intervals[-1] # use end of last interval of last tier - SHOULD USE MAX OF ALL TIERS?
    txt = textgrid_file_intro(xmax, tier_n)
    txt += txt_tiers
    return txt



from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


