import random as rd

keys = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
        'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
        'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
        'C6']
durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]     # 1/16 for minimum duration, 16/16 for maximum duration
phonemes = ['a', 'i', 'u', 'e', 'o',
            'ka', 'ki', 'ku', 'ke', 'ko',
            'sa', 'shi', 'su', 'se', 'so',
            'ta', 'chi', 'tsu', 'te', 'to',
            'na', 'ni', 'nu', 'ne', 'no',
            'ha', 'hi', 'fu', 'he', 'ho',
            'ma', 'mi', 'mu', 'me', 'mo',
            'ya', 'yu', 'yo',
            'ra', 'ri', 'ru', 're', 'ro',
            'wa', 'wo',
            'ga', 'gi', 'gu', 'ge', 'go',
            'za', 'ji', 'zu', 'ze', 'zo',
            'da', 'ji', 'zu', 'de', 'do',
            'ba', 'bi', 'bu', 'be', 'bo',
            'pa', 'pi', 'pu', 'pe', 'po',
            'kya', 'kyu', 'kyo',
            'gya', 'gyu', 'gyo',
            'sha', 'shu', 'sho',
            'ja', 'ju', 'jo',
            'cha', 'chu', 'cho',
            'nya', 'nyu', 'nyo',
            'hya', 'hyu', 'hyo',
            'bya', 'byu', 'byo',
            'pya', 'pyu', 'pyo',
            'mya', 'myu', 'myo',
            'rya', 'ryu', 'ryo',
            'n']


def generate_random_sheets(bars):
    total_beats, current_beats = 16 * bars, 0

    sheets = []

    while current_beats < total_beats:
        key = keys[rd.randrange(0, len(keys), 1)]
        duration = durations[rd.randrange(0, len(durations), 1)]
        phoneme = phonemes[rd.randrange(0, len(phonemes), 1)]
        current_beats += duration
        sheets.append([key, duration, phoneme])

    sheets[-1][1] -= current_beats - total_beats
    
    return sheets

def save_sheets_to_file(filename, bars):
    notes = generate_random_sheets(bars)
    with open(filename, 'w', encoding='utf-8') as file:
        for note in notes:
            file.write(f"{note[0]}, {note[1]}, {note[2]}\n")

save_sheets_to_file("random_sheets.txt", 150)  # Generates for 5 minutes under 120 bpm (4/4)
