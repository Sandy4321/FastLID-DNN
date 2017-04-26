import sys

if len(sys.argv) != 3:
    print "Usage: python label_features.py <features archive file> <utt2lang file>"
    sys.exit(1)

features_ark_filename = sys.argv[1]
features_filename = "%s_labeled" % features_ark_filename[:-4]   # Cuts off .ark from end of file

# Populate utterance-to-language dictionary for quick lookup when loading features
utt2lang_filename = sys.argv[2]
utt2lang_file = open(utt2lang_filename, 'r')
utt2lang = dict()
for line in utt2lang_file.readlines():
    utt, lang = line.strip('\n').split(" ")
    utt2lang[utt] = lang
utt2lang_file.close()

# Write features on each line as "utterance language feat1 feat2 ... feat N"
with open(features_ark_filename, 'r') as features_ark:
    with open(features_filename, 'w') as features:
        current_utt = ""
        current_lang = ""
        for features_ark_line in features_ark.readlines():
            if "[" in features_ark_line:
                # Found a new utterance!
                current_utt = features_ark_line[:4]
                current_lang = utt2lang[current_utt]
            else:
                feature_vec = map(float, features_ark_line.split(" ")[2:-1])
                feature_vec_str = " ".join(map(str, feature_vec))
                features.write("%s %s %s\n" % (current_utt, current_lang, feature_vec_str))
