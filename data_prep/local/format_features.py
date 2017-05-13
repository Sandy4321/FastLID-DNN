import sys

if len(sys.argv) != 2:
    print "Usage: python format_features.py <features archive file>"
    sys.exit(1)

features_ark_filename = sys.argv[1]
features_filename = "%s_formatted" % features_ark_filename[:-4]   # Cuts off .ark from end of file

# Write features on each line as "feat1 feat2 ... feat N"
with open(features_ark_filename, 'r') as features_ark:
    with open(features_filename, 'w') as features:
        current_utt = ""
        current_lang = ""
        for features_ark_line in features_ark.readlines():
            if "[" in features_ark_line:
                # Found a new utterance!
                current_utt = features_ark_line[:4]
            else:
                feature_vec = map(float, features_ark_line.split(" ")[2:-1])
                feature_vec_str = " ".join(map(str, feature_vec))
                features.write("%s\n" % feature_vec_str)
