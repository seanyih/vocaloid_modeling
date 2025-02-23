import os
import soundfile as sf

# Source and destination directories.
src_dir = '/home/fu/Documents/vocaloid_modeling/vocaloid_modeling/vocaloid_modeling/data/audio'
dst_dir = '/home/fu/Documents/vocaloid_modeling/vocaloid_modeling/vocaloid_modeling/data/audio_25s'
os.makedirs(dst_dir, exist_ok=True)
# Duration in seconds to extract.
duration = 25

# List all files in the source directory.
original_files = os.listdir(src_dir)

for file in original_files:
    # Create the full source file path.
    src_path = os.path.join(src_dir, file)
    
    # Read the audio file.
    data, samplerate = sf.read(src_path)
    
    # Calculate the number of samples for the first 5 seconds.
    num_samples = duration * samplerate
    
    # Slice the array to get the first 5 seconds.
    data_25s = data[:num_samples]
    
    # Create the full destination file path.
    dst_path = os.path.join(dst_dir, file)
    
    # Write the 5-second clip to the destination.
    sf.write(dst_path, data_25s, samplerate)
    
    print(f'Processed {file}: saved first 25 seconds to {dst_path}')
