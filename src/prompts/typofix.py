from prompts.prompt_formatting import FewShotPrompt, ChatIO, cut_ends

# Prompt Template
sys_prompt_template = cut_ends('''
Your role is to fix typos. For every typo/list of typos made that is given, you are to reproduce that list in the same order as was given to you, but with the fixed versions. If a term is correct, leave it as is. Also, for each item in the list you produce, you must surround it with ticks (`).

Here's what constitutes as a typo and how they should be fixed:
- Misspelled words -> Fix: spell them correctly
- Anything that includes nonsensical characters (such as periods and unneeded hyphens/dashes) -> Fix: remove them
- Weird or odd spacing -> Fix: fix the spacing to make more sense 
- If it includes a hyphen to denote 'definition' (such as "Topic - subtopic") rather than being part of the word (such as "editor-in-chief") -> Fix: use colons such as "Topic: subtopic" instead of "Topic - subtopic"

Each item will be labeled with ticks (`) so that you know which item is which.
''')

# Few-Shot Examples
_few_shot_1i = cut_ends('''
Before:
[
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
]
''')

_few_shot_1o = cut_ends('''
After:
[
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
]
''')

_few_shot_2i = cut_ends('''
Before:
[
`sight seerer`
`skier -nordic`
`reservore`
`watr`
`truk`
`ice..fishing`
`child$$$4-6.`
`stranded_`
]
''')

_few_shot_2o = cut_ends('''
After:
[
`sightseer`
`skier: nordic`
`reservoir`
`water`
`truck`
`ice fishing`
`child 4-6`
`stranded`
]
''')

_few_shot_3i = cut_ends('''
Before:
[
`potata`
`kuwi`
`hamewark`
`phzsicz`
`fomula ine`
]
''')

_few_shot_3o = cut_ends('''
After:
[
`potato`
`kiwi`
`homework`
`physics`
`formula one`
]
''')

_few_shot_4i = cut_ends('''
Before:
[
`act1v..ati on$`
]
''')

_few_shot_4o = cut_ends('''
After:
[
`activation`
]
''')

# Putting the Few-Shot Examples all together
few_shot_examples = [
    ChatIO(_few_shot_1i, _few_shot_1o),
    ChatIO(_few_shot_2i, _few_shot_2o),
    ChatIO(_few_shot_3i, _few_shot_3o),
    ChatIO(_few_shot_4i, _few_shot_4o)
]

# PUT EVERYTHING TOGETHER
FEW_SHOT_PROMPT = FewShotPrompt(sys_prompt_template, few_shot_examples)




# Text Completion Prompt
cut_ends("""
Your role is to fix typos. For every typo/list of typos made that is given, you are to reproduce that list in the same order as was given to you, but with the fixed versions. If a term is correct, leave it as is.

Here's what constitutes as a typo and how they should be fixed:
- Misspelled words -> Fix: spell them correctly
- Anything that includes nonsensical characters (such as periods and unneeded hyphens/dashes) -> Fix: remove them
- Weird or odd spacing -> Fix: fix the spacing to make more sense 
- If it includes a hyphen to denote 'definition' (such as "Topic - subtopic") rather than being part of the word (such as "editor-in-chief") -> Fix: use colons such as "Topic: subtopic" instead of "Topic - subtopic"

When providing each example, I will separate each of them with ticks (`) So that you know which example is which.

Before:
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

After:
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

Now, try it with this:
Before:
`sight seerer`
`skier -nordic`
`reservore`
`watr`
`truk`
`ice..fishing`
`child$$$4-6.`
`stranded_`

After:
""")