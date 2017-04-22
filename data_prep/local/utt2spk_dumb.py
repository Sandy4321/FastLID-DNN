import sys

if len(sys.argv) != 2:
    print "Usage: python utt2spk_dumb.py <data directory root>"
    sys.exit(1)

data_root = sys.argv[1]
wav_scp_filename = "%s/wav.scp" % data_root
utt2spk_filename = "%s/utt2spk" % data_root

# This is "dumb" because we do not have speaker information: we simply make a
# 1-to-1 mapping of utterance ID to "speaker"
utt2spk_lines = []
with open(wav_scp_filename, 'r') as wav_scp:
    for wav in wav_scp.readlines():
        utt = wav.split(" ")[0]
        utt2spk_lines.append("%s %s\n" % (utt, utt))

# Sort on "speaker"
utt2spk_lines.sort(key=lambda x: x.strip('\n').split(" ")[-1])
with open(utt2spk_filename, 'w') as utt2spk:
    for line in utt2spk_lines:
        utt2spk.write(line)
