


from .praat_ifc import split_by_char_unless_quoted, rename_tier_by_spec, rename_prune_tiers


def test_split_by_char_unless_quoted():
    assert split_by_char_unless_quoted("a:b")==['a', 'b']
    assert split_by_char_unless_quoted("a")==['a']
    assert split_by_char_unless_quoted("abc", split_char='b')==['a', 'c']
    assert split_by_char_unless_quoted("abcd", split_char='c', escape_char='b')==['acd']
    assert split_by_char_unless_quoted("abcd:")==['abcd', '']
    assert split_by_char_unless_quoted(":xyz")==['', 'xyz']
    assert split_by_char_unless_quoted("abc:defg::hij")==['abc', 'defg', '', 'hij']
    assert split_by_char_unless_quoted("")==['']
    assert split_by_char_unless_quoted(":")==['', '']
    assert split_by_char_unless_quoted("::")==['', '', '']
    assert split_by_char_unless_quoted("a\\:b")==['a:b']
    assert split_by_char_unless_quoted("a\\:\\:b")==['a::b']
    assert split_by_char_unless_quoted("a:\\:\\:b")==['a', '::b']


def test_rename_tier_by_spec():
    assert rename_tier_by_spec('tier_name', [])==None
    assert rename_tier_by_spec('old', ['old:new'])=='new'
    assert rename_tier_by_spec('old', [':suffix'])=='oldsuffix'
    assert rename_tier_by_spec('unchanged', [':'])=='unchanged'
    assert rename_tier_by_spec('strange:name', ['strange\\:name:better_name'])=='better_name'
    assert rename_tier_by_spec('old', ['abc:def'])==None
    assert rename_tier_by_spec('old', ['abc:def', ':'])=='old'
    assert rename_tier_by_spec('old', ['abc:def', 'efg', 'old:xxx', 'old:yyy'])=='xxx'


def test_rename_prune_tiers():
    assert rename_prune_tiers({'abc':[(0, 1, 'a'), (1, 2, 'b')]}, ['abc:def'])=={'def':[(0, 1, 'a'), (1, 2, 'b')]}
    assert rename_prune_tiers({'abc':[], 'def':[]}, ['def:xxx'])=={'xxx':[]}
    assert rename_prune_tiers({'abc':[], 'def':[]}, [':'])=={'abc':[], 'def':[]}
    assert rename_prune_tiers({'abc':[], 'def':[(0, 1, 'a')]}, ['x:y', 'def'])=={'def':[(0, 1, 'a')]}
    assert rename_prune_tiers({'abc':[], 'def':[(0, 1, 'a')]}, ['x:y', ':_suf'])=={'abc_suf':[], 'def_suf':[(0, 1, 'a')]}
    assert rename_prune_tiers({'abc':[], 'def':[(0, 1, 'a')]}, ['abc:def', 'def:abc'])=={'def':[], 'abc':[(0, 1, 'a')]}
    assert rename_prune_tiers({'abc:x':[(1, 2, 'b')], 'def':[]}, ['abc\\:x:abc_x', 'def:xxx'])=={'abc_x':[(1, 2, 'b')], 'xxx':[]}
    assert rename_prune_tiers({'abc':[], 'def':[]}, [])=={}
    assert rename_prune_tiers({'abc':[], 'def':[]}, ['zzz'])=={}


