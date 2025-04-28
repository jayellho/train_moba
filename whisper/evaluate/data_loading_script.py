import os
import glob
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import textgrids
import soundfile as sf
import re
import json
import tempfile
import random

_DESCRIPTION = """\
PART 4 contains code-switching
The National Speech Corpus (NSC) is the first large-scale Singapore English corpus 
spearheaded by the Info-communications and Media Development Authority (IMDA) of Singapore.

Summary of Part 4 data organisation:
- Codeswitching
   - Same Room environment, files organized by speaker number:
    /Scripts Same Room: Orthographic transcripts saved in TextGrid format
    /Audio Same Room: Audio files in WAV format recorded using the mobile phone mic, sampled at 16kHz
   - Different Room environment, files organized by speaker number and session number:
    /Scripts Diff Room: Orthographic transcripts saved in TextGrid format 
    /Audio Diff Room: Audio files in WAV format recorded using the mobile phone, sampled at 16kHz
"""

_CITATION = """\
"""

_CHANNEL_CONFIGS = sorted([
    "Diff Room Audio", "Same Room Audio"
])

_GENDER_CONFIGS = sorted(["F", "M"])

# _RACE_CONFIGS = sorted(["CHINESE", "MALAY", "INDIAN", "OTHERS"])
_RACE_CONFIGS = sorted(["CHINESE", "INDIAN"]) #"MALAY",

_HOMEPAGE = "https://www.imda.gov.sg/how-we-can-help/national-speech-corpus"

_LICENSE = ""

# _PATH_TO_DATA = '/media/vest1/SecureUSB/IMDA - National Speech Corpus/PART3'
# _PATH_TO_DATA = './PART1/DATA'
_PATH_TO_DATA = './evaluation_data'

# set the maximum length of spliced clips
INTERVAL_MAX_LENGTH = 25

'''
Function to remove annotations and punctuations in text. 
Accepts a string and outputs a string after formmatting.
'''
def cleanup_string(line):

    words_to_remove = ['(ppo)','(ppc)', '(ppb)', '(ppl)', '<s/>','<c/>','<q/>', '<fil/>', '<sta/>', '<nps/>', '<spk/>', '<non/>', '<unk>', '<s>', '<z>', '<nen>',
                        '<p1>','</p1>', '<p2>', '</p2>', '<p3>','</p3>', '<ex2>','</ex2>']

    formatted_line = re.sub(r'\s+', ' ', line).strip().lower()

    #detect all word that matches words in the words_to_remove list
    for word in words_to_remove:
        if re.search(word,formatted_line):
            # formatted_line = re.sub(word,'', formatted_line)
            formatted_line = formatted_line.replace(word,'')
            formatted_line = re.sub(r'\s+', ' ', formatted_line).strip().lower()
            # print("*** removed words: " + formatted_line)

    #detect '\[(.*?)\].' e.g. 'Okay [ah], why did I gamble?'
    #remove [ ] and keep text within
    if re.search('\[(.*?)\]', formatted_line):
        formatted_line = re.sub('\[(.*?)\]', r'\1', formatted_line).strip()
        #print("***: " + formatted_line)

    #detect '\((.*?)\).' e.g. 'Okay (um), why did I gamble?'
    #remove ( ) and keep text within
    if re.search('\((.*?)\)', formatted_line):
        formatted_line = re.sub('\((.*?)\)', r'\1', formatted_line).strip()
        # print("***: " + formatted_line)

    #detect '\'(.*?)\'' e.g. 'not 'hot' per se'
    #remove ' ' and keep text within
    if re.search('\'(.*?)\'', formatted_line):
        formatted_line = re.sub('\'(.*?)\'', r'\1', formatted_line).strip()
        #print("***: " + formatted_line)

    #remove punctation '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    punctuation = '''!–;",.?@#$%^&*~''' # don't remove the "/" and "\"
    punctuation_list = str.maketrans("","",punctuation)
    formatted_line = re.sub(r'-', ' ', formatted_line)
    formatted_line = re.sub(r'_', ' ', formatted_line)
    formatted_line = formatted_line.translate(punctuation_list)
    formatted_line = re.sub(r'\s+', ' ', formatted_line).strip().lower()
    #print("***: " + formatted_line)

    return formatted_line

def clean_locale(line):
    sent = ''
    line = line.lower()
    cleaned_phrase = ''
    while re.search(r"<.*?>|</.*?>", line):
        match = re.search(r"<([^<>]+)>", line)
        # print('match:',match)
        start_line = line[:match.span()[0]]
        # print('start line:',start_line)
        end_match = re.search(r"</([^<>]+)>", line)
        end_line = line[end_match.span()[1]:]
        # print('end line:',end_line)
        if match:
            locale = match.group()
            # print('locale:', locale)

            if locale == "<mandarin>":
                tmpCN = re.findall(r"<mandarin>(.*?)<\/mandarin>", line)
                # print('tmp',tmpCN)
                cleaned_phrase = re.sub(r":.*$", "", tmpCN[0])
                # print(cleaned_phrase)
            
            if locale == "<malay>":
                tmpCN = re.findall(r"<malay>(.*?)<\/malay>", line)
                # print('tmp',tmpCN)
                cleaned_phrase = re.sub(r":.*$", "", tmpCN[0])
                # print(cleaned_phrase)
            
            if locale == "<tamil>":
                tmpCN = re.findall(r"<tamil>(.*?)<\/tamil>", line)
                # print('tmp',tmpCN)
                cleaned_phrase = re.sub(r"^(.*?):", "", tmpCN[0])
                # print(cleaned_phrase)
            
            try:
                sent = start_line + cleaned_phrase + end_line
                line = sent
            except Exception as e:
                print(e)
        else:
            continue

    sent = line
    return sent

def read_textgrid(script_path):
    # read the textgrid into tg, using python package praat-textgrids.
    try:
        # try utf-8
        with open(script_path, "rb") as f:
            tg = f.read()
            tg_dict = textgrids.TextGrid() 
            tg_dict.parse(tg)
        for key in tg_dict.keys():
            tg = tg_dict[key]
        
        return tg
    except UnicodeDecodeError:
        # try utf-16
        try:
            with open(script_path, "rb") as f:
                tg = f.read()
                decoded = tg.decode('utf-16')
                encoded = decoded.encode('utf-8')
                tg_dict = textgrids.TextGrid() 
                tg_dict.parse(encoded)
            for key in tg_dict.keys():
                tg = tg_dict[key]
            
            return tg
        except Exception as e:
            print(f"error reading textgrid file, {script_path}, {str(e)}")
           
    except TypeError:
        # try utf-8-sig
        try:
            with open(script_path, "rb") as f:
                tg = f.read()
                decoded = tg.decode('utf-8-sig')
                encoded = decoded.encode('utf-8')
                tg_dict = textgrids.TextGrid() 
                tg_dict.parse(encoded)
            for key in tg_dict.keys():
                tg = tg_dict[key]
            
            return tg
        except Exception as e:
            print(f"error reading textgrid file, {script_path}, {str(e)}")
            
    except Exception as e:
        print(f"error reading textgrid file, {script_path}, {str(e)}")

  



class PART4Config(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(
        self, channel, gender, race, description, homepage, path_to_data
    ):
        super(PART4Config, self).__init__(
            name=channel+gender+race,
            version=datasets.Version("1.0.0", ""),
            description=self.description,
        )
        self.channel = channel
        self.gender = gender
        self.race = race
        self.description = description
        self.homepage = homepage
        self.path_to_data = path_to_data


def _build_config(channel, gender, race):
    return PART4Config(
        channel=channel,
        gender=gender,
        race=race,
        description=_DESCRIPTION,
        homepage=_HOMEPAGE,
        path_to_data=_PATH_TO_DATA,
    )

class PART4Dataset(datasets.GeneratorBasedBuilder):
    """
    This dataset returns a spliced 20-25 second audio clip together with the transcript, channel, gender, race, name of the audio fileand the time interval.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = []
    for channel in _CHANNEL_CONFIGS + ["all"]:
        for gender in _GENDER_CONFIGS + ["all"]:
            for race in _RACE_CONFIGS + ["all"]:
                BUILDER_CONFIGS.append(_build_config(channel,gender,race))

    DEFAULT_CONFIG_NAME = "allallall"

    def _info(self):
        features = datasets.Features(
            {
                "audio": datasets.features.Audio(sampling_rate=16000),
                "transcript": datasets.Value("string"),
                "mic": datasets.Value("string"),
                "audio_name": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "race": datasets.Value("string"),
                "interval": datasets.Value("string")
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("audio", "transcript"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            # task_templates=None,
        )

    def _split_generators(self, dl_manager):
        mics = (
            _CHANNEL_CONFIGS
            if self.config.channel == "all"
            else [self.config.channel]
        )

        gender = (
            _GENDER_CONFIGS
            if self.config.gender == "all"
            else [self.config.gender]
        )

        race = (
            _RACE_CONFIGS
            if self.config.race == "all"
            else [self.config.race]
        )

        # read in speaker metadata from the dataset 
        path_to_speaker = os.path.join(self.config.path_to_data, "NSC Part 4 Speaker Metadata.xlsx")
        print(f"path to speaker",path_to_speaker)
        speaker_df = pd.read_excel(path_to_speaker, dtype={'Speaker ID': object}) # read in speaker ID as a string rather than integers

        # train-test split the speakers within their gender and race type (to ensure matching proportions)
        # train_speaker_ids = []
        # test_speaker_ids = []
        # for g in gender:
        #     for r in race:
        #         X = speaker_df[(speaker_df["Ethnic Group"]==r) & (speaker_df["Gender"]==g)]
        #         # print("X {}".format(X))
        #         if len(X) == 0:
        #             continue
        #         X_train, X_test = train_test_split(X, test_size=1, random_state=42, shuffle=True)
        #         train_speaker_ids.extend(X_train["Speaker ID"])
        #         test_speaker_ids.extend(X_test["Speaker ID"])
        all_speaker_ids = speaker_df["Speaker ID"].tolist()
        train_speaker_ids = all_speaker_ids.copy()
        test_speaker_ids  = all_speaker_ids.copy()
        print("TRAIN: {}".format(train_speaker_ids))
        print("TEST: {}".format(test_speaker_ids))

        return [
            datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "path_to_data": self.config.path_to_data,
                "speaker_metadata":speaker_df,
                "speaker_ids": train_speaker_ids,
                # "speaker_ids":["0001"],
                "mics": tuple(mics),
                "gender": tuple(gender),
                "race":tuple(race),
                "dl_manager": dl_manager
              },
          ),
          datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "path_to_data": self.config.path_to_data,
                "speaker_metadata":speaker_df,
                "speaker_ids": test_speaker_ids,
                # "speaker_ids": ["0003"],
                "mics": mics,
                "gender": gender,
                "race":race,
                "dl_manager": dl_manager
            },
        ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
            self,
            path_to_data,
            speaker_metadata,
            speaker_ids,
            mics,
            gender,
            race,
            dl_manager
        ):
        id_ = 0  # for each data point generated 
        s1= 0
        for mic in mics:
            audio_path = ''
            room = " ".join(mic.split(" ")[:2]) + " Scripts"
            print("ROOM NAME: {}".format(room))
            for speaker in speaker_ids:
                print('SPEAKERIDS', len(speaker_ids))
                print('s1',s1)
                print('speakerid1 {}'.format(speaker))
                if speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Ethnic Group'].values[0] == 'CHINESE':
                    session_id = str(speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Session ID'].values[0])
                    session_id = session_id.zfill(4)
                    d = {}
                    if mic == 'Diff Room Audio':
                        try:
                            audio_path = os.path.join(path_to_data, "Codeswitching", mic, 'sur_' + str(session_id) + '_' + str(speaker) + '_phnd_cs-chn.wav')
                            filename = 'sur_' + str(session_id) + '_' + str(speaker) + '_phnd_cs-chn.TextGrid'
                            script_path = dl_manager.download(os.path.join(path_to_data, "Codeswitching", room, filename))

                            if os.path.exists(script_path):
                                tg = read_textgrid(script_path)

                        except Exception as e:
                            # print(f"error getting script path, {str(e)}")
                            continue

                    elif mic == 'Same Room Audio':
                        try:
                            audio_path = os.path.join(path_to_data, "Codeswitching", mic, 'sur_' + str(session_id) + '_' + str(speaker) + '_phns_cs-chn.wav')
                            filename = 'sur_' + str(session_id) + '_' + str(speaker) + '_phns_cs-chn.TextGrid'
                            script_path = dl_manager.download(os.path.join(path_to_data, "Codeswitching", room, filename))

                            if os.path.exists(script_path):
                                tg = read_textgrid(script_path)

                        except Exception as e:
                            # print(f"error getting script path, {str(e)}") #maybe can add count to check if the total number of audio is correct 
                            continue
                
                elif speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Ethnic Group'].values[0] == 'MALAY':
                    session_id = str(speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Session ID'].values[0])
                    session_id = session_id.zfill(4)
                    d = {}
                    if mic == 'Diff Room Audio':
                        try:
                            audio_path = os.path.join(path_to_data, "Codeswitching", mic, 'sur_' + str(session_id) + '_' + str(speaker) + '_phnd_cs-mly.wav')
                            filename = 'sur_' + str(session_id) + '_' + str(speaker) + '_phnd_cs-mly.TextGrid'
                            script_path = dl_manager.download(os.path.join(path_to_data, "Codeswitching", room, filename))

                            if os.path.exists(script_path):
                                tg = read_textgrid(script_path)

                        except Exception as e:
                            # print(f"error getting script path, {str(e)}")
                            continue

                    elif mic == 'Same Room Audio':
                        try:
                            audio_path = os.path.join(path_to_data, "Codeswitching", mic, 'sur_' + str(session_id) + '_' + str(speaker) + '_phns_cs-mly.wav')
                            filename = 'sur_' + str(session_id) + '_' + str(speaker) + '_phns_cs-mly.TextGrid'
                            script_path = dl_manager.download(os.path.join(path_to_data, "Codeswitching", room, filename))
                            if os.path.exists(script_path):
                                tg = read_textgrid(script_path)
                        except Exception as e:
                            # print(f"error getting script path, {str(e)}") 
                            continue
            
                            
                elif speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Ethnic Group'].values[0] == 'INDIAN' or speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Ethnic Group'].values[0] == 'OTHERS':
                    # Note: In PART4, the files of others ends with cs-tml which refers to Indians
                    session_id = str(speaker_metadata[speaker_metadata['Speaker ID']==speaker]['Session ID'].values[0])
                    session_id = session_id.zfill(4)
                    d = {}
                    if mic == 'Diff Room Audio':
                        try:
                            audio_path = os.path.join(path_to_data, "Codeswitching", mic, 'sur_' + str(session_id) + '_' + str(speaker) + '_phnd_cs-tml.wav')
                            filename = 'sur_' + str(session_id) + '_' + str(speaker) + '_phnd_cs-tml.TextGrid'
                            script_path = dl_manager.download(os.path.join(path_to_data, "Codeswitching", room, filename))

                            if os.path.exists(script_path):
                                tg = read_textgrid(script_path)

                        except Exception as e:
                            # print(f"error getting script path, {str(e)}")
                            continue

                    elif mic == 'Same Room Audio':
                        try:
                            audio_path = os.path.join(path_to_data, "Codeswitching", mic, 'sur_' + str(session_id) + '_' + str(speaker) + '_phns_cs-tml.wav')
                            filename = 'sur_' + str(session_id) + '_' + str(speaker) + '_phns_cs-tml.TextGrid'
                            script_path = dl_manager.download(os.path.join(path_to_data, "Codeswitching", room, filename))

                            if os.path.exists(script_path):
                                tg = read_textgrid(script_path)

                        except Exception as e:
                            # print(f"error getting script path, {str(e)}") #maybe can add count to check if the total number of audio is correct 
                            continue
                    
                else:
                    print("Ethnic Group Doesn't Exist")
                    print('SPEAKER ID IS {}'.format(speaker))

                # check that audio path exists, else will not open the audio
                if os.path.exists(audio_path):
                    try:
                        # read the audio file with soundfile
                        with open(audio_path, 'rb') as f:
                            data, sr = sf.read(f)
                            # skip the audio file if the sampling rate is not 16000
                            # if sr != 16000:
                            #     print(f'sample rate: {sr}')
                            #     continue

                        # initialize variables
                        result = {} # initialize the result dictionary to generate each spliced clip
                        i = 0 # initialize the counter for textgrid transcripts
                        intervalLength = 0 # initialize a variable to record the duration of the current interval to be spliced
                        intervalStart = 0 # initialize a variable to record the start time of the current interval to be spliced
                        transcript_list = [] # initialize the list of the transcripts added together for one spliced clip
                        
                        # create a temporary file path to store the spliced clip
                        tempWavFile = tempfile.mktemp('.wav')

                        # loop through the list of textgrid intervals
                        while i < (len(tg)-1):
                            # process the transcript text
                            transcript = cleanup_string(tg[i].text)
                            transcript = clean_locale(transcript)
                            # if the processed transcript is an empty string, skip to the next textgrid interval
                            if intervalLength == 0 and len(transcript) == 0:
                                intervalStart = tg[i].xmax
                                i+=1
                                continue
                            
                            # if the interval length is greater than the maximum length, we do not use the interval. (because cannot split the text, so drop it)
                            if (tg[i].xmax-tg[i].xmin) > INTERVAL_MAX_LENGTH:
                                print(f"Interval is too long: {tg[i].xmax-tg[i].xmin}")

                                # in the case where this interval is the start of a new clip, we just skip the interval
                                if (tg[i-1].xmax - intervalStart) < 1:
                                    # resetting all variables
                                    intervalLength = 0
                                    intervalStart = tg[i+1].xmin # set to the start time of the next interval
                                    transcript_list = []
                                    i+=1
                                    continue

                                # if it is not the start of a new clip, then we just splice out the clip before this interval.
                                else:
                                    spliced_audio = data[int(intervalStart*sr):int(tg[i-1].xmax*sr)]
                                    sf.write(tempWavFile, spliced_audio, sr)
                                    result["transcript"] = transcript
                                    result["interval"] = "start:"+str(tg[i].xmin)+", end:"+str(tg[i].xmax)
                                    result["audio"] = {"path": tempWavFile, "bytes": spliced_audio, "sampling_rate":sr}
                                    result["audio_name"] = audio_path
                                    result["gender"] = gender
                                    result["race"] = race
                                    yield id_, result
                                    id_+= 1
                                    # resetting all variables
                                    intervalLength=0
                                    intervalStart=tg[i+1].xmin # set to the start time of the next interval
                                    transcript_list = []

                            # in most cases where the interval is less than the maximum length
                            else:
                                # add the interval duration to the current interval length
                                intervalLength += tg[i].xmax-tg[i].xmin

                                # if adding the next interval will not exceed the maximum length, then we add the text to the transcript list, and move to the next interval.
                                if (intervalLength + tg[i+1].xmax-tg[i+1].xmin) < INTERVAL_MAX_LENGTH:
                                    if len(transcript) != 0:
                                        transcript_list.append(transcript)
                                    i+=1
                                    continue

                                # else if adding the next interval will exceed the maximum length, then we splice out the audio clip up until this interval.
                                if len(transcript) == 0: # in the case where the current transcript is an empty string, then we do not include this interval.
                                    spliced_audio = data[int(intervalStart*sr):int(tg[i].xmax*sr)]
                                else: # in most cases, we will add the current transcript.
                                    transcript_list.append(transcript)
                                    spliced_audio = data[int(intervalStart*sr):int(tg[i].xmax*sr)]
                                
                                # splice out the clip and generate a data point in the dataset.
                                sf.write(tempWavFile,spliced_audio, sr )
                                result["interval"] = "start:"+str(intervalStart)+", end:"+str(tg[i].xmax)
                                result["audio"] = {"path": tempWavFile, "bytes": spliced_audio, "sampling_rate":sr}
                                result["transcript"] = ' '.join(transcript_list)
                                result["audio_name"] = audio_path
                                result["gender"] = gender
                                result["race"] = race
                                yield id_, result
                                id_+= 1
                                # resetting all variables
                                intervalLength=0
                                intervalStart=tg[i+1].xmin # set to the start time of the next interval
                                transcript_list = []
                                
                            i+=1
                    except Exception as e:
                        print(e)
                        continue
                s1 += 1
        print("\nCOMPLETED")