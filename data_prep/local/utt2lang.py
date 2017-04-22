import sys

if len(sys.argv) != 3:
    print "Usage: python utt2lang.py <data directory root> <language seg ndx file>"
    sys.exit(1)

data_root = sys.argv[1]
wav_scp_filename = "%s/wav.scp" % data_root
utt2lang_filename = "%s/utt2lang_unsorted" % data_root
languages_filename = "%s/languages" % data_root
seg_lang_ndx_filename = sys.argv[2]

utt2lang_lines = []
with open(wav_scp_filename, 'r') as wav_scp:
    with open(seg_lang_ndx_filename, 'r') as seg_lang_ndx:
        with open(languages_filename, 'r') as languages:
            with open(utt2lang_filename, 'w') as utt2lang:
                # Should have same number of mappings
                wavs = wav_scp.readlines()
                seg_langs = seg_lang_ndx.readlines()
                assert(len(wavs) == len(seg_langs))

                languages_list = map(lambda x: x.strip('\n').lower(), languages.readlines())

                for i in xrange(len(wavs)):
                    wav = wavs[i]
                    utt = wav.split(" ")[0]
                    seg_lang = seg_langs[i].strip('\n').split(" ")[-1]

                    # Check if language is out-of-set
                    if seg_lang not in languages_list:
                        seg_lang = "oos"
                    
                    utt2lang.write("%s %s\n" % (utt, seg_lang))
