import math
import io
import re
import os

import numpy as np
from scipy.io import wavfile

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums, types

# Sets authentication environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'auth/Chemistry-NU.json'

# Instantiates a client
client = speech.SpeechClient()

def gcs(signal, fs):
    audio = types.RecognitionAudio(content=signal.tobytes())

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fs,
        language_code='en-US',
        max_alternatives=1,
        model='video')
        #enable_automatic_punctuation=True,
        #audio_channel_count=6,
        #enable_separate_recognition_per_channel=True)
        
    # Detects speech in the audio file
    response = client.recognize(config, audio)

    return response

# Word Error Rate calculation based on jiwer: https://github.com/jitsi/asr-wer
def wer(human_transcript, asr_transcript):
    vocab = []

    for word in human_transcript + asr_transcript:
        if word not in vocab:
            vocab.append(word)
    
    # Represent transcripts by index (as integers)
    h = []
    a = []

    for word in human_transcript:
        h.append(vocab.index(word))

    for word in asr_transcript:
        a.append(vocab.index(word))

    # Alignment
    distance = _edit_distance(h, a)

    # Calculate WER from edit distance
    n = len(human_transcript)
    word_error_rate = distance / n

    return word_error_rate

def _edit_distance(a, b):
    # Calculate edit distance based on Wagner-Fischer algorithm
    if len(a) == 0:
        raise ValueError('Reference string cannot be empty.')
    elif len(b) == 0:
        return len(a)

    # Initialize matrix and set the first row and column equal to 1, 2, 3, ...
    # Each column represents a single token in the reference string a
    # Each row represents a single token in the reference string b
    m = np.zeros((len(b) + 1, len(a) + 1), dtype=np.int32)

    m[0, 1:] = np.arange(1, len(a) + 1)
    m[1:, 0] = np.arange(1, len(b) + 1)

    # Loop over remaining cells (from second row and column onwards)
    # The value of each selected cell is:
    #   if token represented by row == token represented by column:
    #       value of the top-left diagonal cell
    #   else:
    #       calculate 3 values:
    #            * top-left diagonal cell + 1 (which represents substitution)
    #            * left cell + 1 (representing deleting)
    #            * top cell + 1 (representing insertion)
    #       value of the smallest of the three
    for i in range(1, m.shape[0]):
        for j in range(1, m.shape[1]):
            if a[j-1] == b[i-1]:
                m[i, j] = m[i-1, j-1]
            else:
                m[i, j] = min(
                    m[i-1, j-1] + 1,
                    m[i, j-1] + 1,
                    m[i-1, j] + 1
                )

    # The minimum-edit distance is the value of the bottom-right cell of matrix
    return m[len(b), len(a)]

# Load audio recording
fs, recording = wavfile.read('audio/Sony In lab Audio/32.wav')
recording = recording[:,0] # Grab 1st channel

# recording = recording[:,:6] # Grab 6 channels
# combined_channels = np.zeros(len(recording), dtype=np.int16)
# for col in range(recording.shape[1]):
#     combined_channels += recording[:,col]
# combined_channels //= 6
# recording = combined_channels

# Read human transcript
with open('audio/Sony In lab Audio/32.txt', 'r') as f:
    timestamps = []
    utterances = []
    for line in f.readlines():
        timestamp = line[1:9]
        line = re.sub('\[.*?\]', ' ', line) # remove timestamps and inaudible tags
        try:
            line = line[line.index(':'):]
        except ValueError:
            continue
        line = line.replace('...', ' ') # replace ellipses with spaces
        line = line.translate(str.maketrans('', '', '!"#&()*+,./:;<=>?@[\\]^_`{|}~')) # remove punctuation
        line = re.sub(' +', ' ', line) # replace double spaces with single spaces
        line = line.strip('\n ') # remove newline character and leading/trailing spaces
        timestamps.append(timestamp)
        utterances.append(line)

# Convert timestamp of each utterance into seconds
seconds = []
for time in timestamps:
    seconds.append(int(time[3:5]) * 60 + int(time[6:8]))

# Convert seconds to frame number
frames = [s * fs for s in seconds]

# Separate audio recording into clips based on utterance with 1 second padding at end
clips = []
for t in range(0, len(frames) - 1):
    if frames[t] == frames[t + 1]: # When two utterances begin at the same time (more than one person talking)
        clips.append(recording[frames[t]:frames[t + 2] + 16000])
    else:
        clips.append(recording[frames[t]:frames[t + 1] + 16000])

# Get GCS responses for each clip
responses = []
for clip_num, clip in enumerate(clips):
    if len(clip) > 960000:
        print(f'Getting response for clip {clip_num}...')
        print(f'Clip {clip_num} longer than 1 minute, splitting into segments...')
        n = math.ceil(len(clip) / 960000)
        segments = np.array_split(clip, n)
        response_segments = []
        for segment in segments:
            response_segment = gcs(segment, fs)
            response_segments.append(response_segment)
        response = response_segments[0]
        for i in range(1, len(response_segments)):
            response.MergeFrom(response_segments[i])
    else:
        print(f'Getting response for clip {clip_num}...')
        response = gcs(clip, fs)
    responses.append(response)

# Get transcripts from GCS responses
transcripts = []
for response in responses:
    transcript = ''
    for result in response.results:
        for alternative in result.alternatives:
            transcript = transcript + ' ' + alternative.transcript
    transcripts.append(transcript)

# Remove double and leading/trailing spaces
transcripts = [re.sub(' +', ' ', transcript) for transcript in transcripts]
transcripts = [transcript.strip() for transcript in transcripts]

# Convert to lower
human_transcripts = [utterance.lower() for utterance in utterances][:-1] # Omit last utterance
google_transcripts = [transcript.lower() for transcript in transcripts]

# Turn strings into list format
human_tokens = [transcript.split() for transcript in human_transcripts]
google_tokens = [transcript.split() for transcript in google_transcripts]

# Calculate WERs and normalize based on word count
total_wc = 0
overall_wer = 0

for h, g in zip(human_tokens, google_tokens):
    total_wc += len(h)
    overall_wer += wer(h, g) * len(h)
    
overall_wer /= total_wc

print(f'WER: {overall_wer}')
print(f'Total WC: {total_wc}')

# Save transcripts and results
with open('transcribed_utterances.txt', 'w') as out:
    for line in human_transcripts:
        out.write(line + '\n')
        
with open('recognized_utterances.txt', 'w') as out:
    for line in google_transcripts:
        out.write(line + '\n')

with open('wer_results.txt', 'w') as out:
    out.write(f'WER: {overall_wer}\n')
    out.write(f'Total WC: {total_wc}')