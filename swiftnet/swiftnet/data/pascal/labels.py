from collections import namedtuple

# Define the namedtuple with added fields
Label = namedtuple('Label', ['name', 'id', 'trainId', 'color', 'ignoreInEval'])

# Define the labels
    #       name          id  trainId  color    ignoreInEval
labels = [
    Label('background',     0, 0, (0, 0, 0), False), 
    Label('aeroplane',      1, 1, (128, 0, 0), False),
    Label('bicycle',        2, 2, (0, 128, 0), False),
    Label('bird',           3, 3, (128, 128, 0), False),
    Label('boat',           4, 4, (0, 0, 128), False),
    Label('bottle',         5, 5, (128, 0, 128), False),
    Label('bus',            6, 6, (0, 128, 128), False),
    Label('car',            7, 7, (128, 128, 128), False),
    Label('cat',            8, 8, (64, 0, 0), False),
    Label('chair',          9, 9, (192, 0, 0), False),
    Label('cow',            10, 10, (64, 128, 0), False),
    Label('dining table',   11, 11, (192, 128, 0), False),
    Label('dog',            12, 12, (64, 0, 128), False),
    Label('horse',          13, 13, (192, 0, 128), False),
    Label('motorbike',      14, 14, (64, 128, 128), False),
    Label('person',         15, 15, (192, 128, 128), False),
    Label('potted plant',   16, 16, (0, 64, 0), False),
    Label('sheep',          17, 17, (128, 64, 0), False),
    Label('sofa',           18, 18, (0, 192, 0), False),
    Label('train',          19, 19, (128, 192, 0), False),
    Label('tv/monitor',     20, 20, (0, 64, 128), False)
]