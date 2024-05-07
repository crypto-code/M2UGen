#############################################################################
# Set the path
raw_path = '../data/raw/'
checkpoint_path = '../data/checkpoint/'
output_path = '../data/Looperman_Loops/'

#############################################################################
# Set the target URL for the web scraping
categories = [
    {'name': 'Bass', 'num': 4356, 'cid': 2, 'max_page': 175},
    {'name': 'Bass Guitar', 'num': 1301, 'cid': 43, 'max_page': 53},
    {'name': 'Bass Synth', 'num': 5234, 'cid': 44, 'max_page': 210},
    {'name': 'Bass Wobble', 'num': 2832, 'cid': 39, 'max_page': 114},
    {'name': 'Drum', 'num': 41327, 'cid': 1, 'max_page': 1654},
    {'name': 'Flute', 'num': 4825, 'cid': 7, 'max_page': 193},
    {'name': 'Guitar Acoustic', 'num': 16731, 'cid': 33, 'max_page': 670},
    {'name': 'Guitar Electric', 'num': 18461, 'cid': 3, 'max_page': 739},
    {'name': 'Harp', 'num': 1315, 'cid': 41, 'max_page': 53},
    {'name': 'Percussion', 'num': 3241, 'cid': 20, 'max_page': 130},
    {'name': 'Piano', 'num': 36948, 'cid': 21, 'max_page': 1478},
    {'name': 'Scratch', 'num': 174, 'cid': 12, 'max_page': 7},
    {'name': 'Strings', 'num': 6077, 'cid': 10, 'max_page': 244},
    {'name': 'Synth', 'num': 47470, 'cid': 4, 'max_page': 1899},
    {'name': 'Violin', 'num': 1280, 'cid': 29, 'max_page': 52}
]

# sorted_categories = sorted(categories, key=lambda x:x['num'])
sorted_categories = [
    {'name': 'Scratch', 'num': 174, 'cid': 12, 'max_page': 7},
    {'name': 'Violin', 'num': 1280, 'cid': 29, 'max_page': 52},
    {'name': 'Bass Guitar', 'num': 1301, 'cid': 43, 'max_page': 53},
    {'name': 'Harp', 'num': 1315, 'cid': 41, 'max_page': 53},
    {'name': 'Bass Wobble', 'num': 2832, 'cid': 39, 'max_page': 114},
    {'name': 'Percussion', 'num': 3241, 'cid': 20, 'max_page': 130},
    {'name': 'Bass', 'num': 4356, 'cid': 2, 'max_page': 175},
    {'name': 'Flute', 'num': 4825, 'cid': 7, 'max_page': 193},
    {'name': 'Bass Synth', 'num': 5234, 'cid': 44, 'max_page': 210},
    {'name': 'Strings', 'num': 6077, 'cid': 10, 'max_page': 244},
    {'name': 'Guitar Acoustic', 'num': 16731, 'cid': 33, 'max_page': 670},
    {'name': 'Guitar Electric', 'num': 18461, 'cid': 3, 'max_page': 739},
    {'name': 'Piano', 'num': 36948, 'cid': 21, 'max_page': 1478},
    {'name': 'Drum', 'num': 41327, 'cid': 1, 'max_page': 1654},
    {'name': 'Synth', 'num': 47470, 'cid': 4, 'max_page': 1899}
 ]
#############################################################################