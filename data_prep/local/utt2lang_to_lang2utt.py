import sys

if len(sys.argv) != 2:
    print "Usage: python utt2lang_to_lang2utt.py <data directory root>"
    sys.exit(1)

data_root = sys.argv[1]
utt2lang_filename = "%s/utt2lang" % data_root
lang2utt_filename = "%s/lang2utt" % data_root

utt_lists = dict()      # Maps language string to list of utterance strings
with open(utt2lang_filename, 'r') as utt2lang:
    for utt2lang_line in utt2lang.readlines():
        utt, lang = utt2lang_line.strip('\n').split(" ")

        # Check if language is out-of-set
        if lang not in utt_lists:
            utt_lists[lang] = [utt]
        else:
            utt_lists[lang].append(utt)

with open(lang2utt_filename, 'w') as lang2utt:
    for lang in sorted(utt_lists.keys()):
        print "%s: %d utterances for %s" % (lang2utt_filename, len(utt_lists[lang]), lang)
        lang2utt.write("%s %s\n" % (lang, " ".join(utt_lists[lang])))
