from pyroller.hybrid.stitching import stitch_evidence
from pyroller.hybrid.matching import lyric_inventory,select_instances,stitch_windows
from pyroller.hybrid.discovery import discover
from pyroller.hybrid.config import AlignmentSettings
from test_alignment import window,words


def selected(windows,lyrics):
    settings=AlignmentSettings(stitching_mode='evidence');inventory=lyric_inventory(lyrics)
    stream,reports=stitch_evidence(windows,inventory,settings)
    cs,_=discover(stream,inventory,settings)
    path=select_instances(stream,cs,settings)[0]
    return stream,reports,[cs[i] for i in path['candidates']]


def test_no_shared_word_keeps_left_phrase_tail():
    left=window(0,0,10,0,words('i got spirit i got wings',6.,.6))
    right=window(1,6,16,8,words('take me high',12.,.5))
    stream,_,rows=selected([left,right],'I got spirit I got wings\nTake me high')
    assert [r['text'] for r in rows]==['I got spirit I got wings','Take me high']
    assert [w['token'] for w in stream[:6]]=='i got spirit i got wings'.split()


def test_shared_word_does_not_replace_good_suffix_with_bad_one():
    left=window(0,0,10,0,words('just fly for your life',6.,.5))
    right=window(1,6,16,8,words('just like falling in love',6.,.5)+words('take me high',10.,.5))
    _,_,rows=selected([left,right],'Just fly for your life\nTake me high')
    assert [r['text'] for r in rows]==['Just fly for your life','Take me high']


def test_same_occurrence_in_two_windows_not_counted_twice():
    left=window(0,0,10,0,words('take me high',6.,.5))
    right=window(1,6,16,8,words('take me high',6.2,.5))
    _,_,rows=selected([left,right],'Take me high')
    assert len(rows)==1


def test_real_repetitions_survive_boundary():
    left=window(0,0,10,0,words('take me high',4.,.5)+words('take me high',8.,.5))
    right=window(1,6,16,8,words('goodbye now',13.,.5))
    _,_,rows=selected([left,right],'Take me high\nGoodbye now')
    assert [r['text'] for r in rows].count('Take me high')==2


def test_tie_preserves_legacy_and_unknown_words():
    windows=[window(0,0,10,0,words('unknown noise',6.,.5)),window(1,6,16,8,words('other noise',7.,.5))]
    stream,reports,rows=selected(windows,'Take me high')
    old,_=stitch_windows(windows,AlignmentSettings())
    assert [w['token'] for w in stream]==[w['token'] for w in old]
    assert reports[0]['selected_alternative']==0 and rows==[]


def test_preview_only_uses_present_windows_and_empty_window_safe():
    first=window(0,0,10,0,words('take me high',6.,.5))
    stream,_,_=selected([first],'Take me high')
    assert {w['window'] for w in stream}=={0}
    after,_,_=selected([first,window(1,6,16,8,[])],'Take me high')
    assert stream==after


def test_right_window_can_supply_better_phrase():
    left=window(0,0,10,0,words('just like falling in love',6.,.5))
    right=window(1,6,16,8,words('just fly for your life',6.,.5))
    _,_,rows=selected([left,right],'Just fly for your life')
    assert [r['text'] for r in rows]==['Just fly for your life']


def test_long_empty_gap_preserves_prefix_and_new_words():
    windows=[window(0,0,10,0,words('take me high',1.,.5)),
             window(1,8,40,10,[]),
             window(2,60,70,62,words('goodbye now',64.,.5))]
    stream,reports,rows=selected(windows,'Take me high\nGoodbye now')
    assert [r['text'] for r in rows]==['Take me high','Goodbye now']
    assert [w['token'] for w in stream]=='take me high goodbye now'.split()
    assert reports[-1]['method']=='no_recent_words_append'
