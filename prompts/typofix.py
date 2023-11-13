import ChatIO
import cut_ends

# Prompt Template
SYSPROMPT_TEMPLATE = cut_ends('''
Your role is to fix typos. For every typo/list of typos made that is given, you are to reproduce that list in the same order as was given to you, but with the fixed versions. If it is correct, leave it as is.

Here's what constitutes as a typo and how they should be fixed:
- Misspelled words; Fix: spell them correctly
- Anything that includes nonsensical characters (such as periods and unneeded hyphens/dashes); Fix: remove them
- Weird or odd spacing; Fix: fix the spacing to make more sense 
- If it includes a hyphen to denote 'definition' (such as "Topic - subtopic") rather than being part of the word (such as "editor-in-chief"); Fix: use colons such as "Topic: subtopic" instead of "Topic - subtopic"

When providing each example, I will separate each of them with ticks (`) So that you know which example is which
''')

# Few-Shot Examples
_few_shot_1i = cut_ends('''
`catt`
`hunter-`
`huner`
`-alzheimer's patient`
`snowborder`
`snowboarder`
`missing perseon`
`other-camper`
`bicylist`
`dog_`
`aircraft   e-`
`flood***victimss` 
''')

_few_shot_1o = cut_ends('''
`cat`
`hunter`
`hunter`
`alzheimer's patient`
`snowboarder`
`snowboarder`
`missing person`
`other: camper`
`bicyclist`
`dog`
`aircraft`
`flood victims`
''')

_few_shot_2i = cut_ends('''
`sight seerer`
`skier -nordic`
`reservore`
`watr`
`truk`
`ice..fishing`
`child$$$4-6.`
`stranded_`
''')

_few_shot_2o = cut_ends('''
`sightseer`
`skier: nordic`
`reservoir`
`water`
`truck`
`ice fishing`
`child 4-6`
`stranded`
''')

_few_shot_3i = cut_ends('''
`potata`
`kuwi`
`hamewark`
`phzsicz`
`fomula ine`
''')

_few_shot_3o = cut_ends('''
`potato`
`kiwi`
`homework`
`physics`
`formula one`
''')

# Putting the Few-Shot Examples all together
FEW_SHOT_EXAMPLES = [
    ChatIO(_few_shot_1i, _few_shot_1o),
    ChatIO(_few_shot_2i, _few_shot_2o),
    ChatIO(_few_shot_3i, _few_shot_3o)
]

