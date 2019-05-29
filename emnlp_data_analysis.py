import csv, os, re
import nltk
import numpy as np
import random

VIDEO_GAME_SLOTS = ["player_perspective", "genres", "platforms", "esrb", "rating", "exp_release_date", "developer",
                        'has_multiplayer', 'available_on_steam', 'has_linux_release', 'has_mac_release',
                        "name", "specifier", "release_year"]

VIDEO_GAME_DAS = ["confirm", "give_opinion", "recommend", "request", "request_attribute", "request_explanation",
                  "suggest", "verify_attribute", "inform"]

def read_csv(filename, delimiter=",", encoding="utf-8"):
    with open(filename, 'r', encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile,delimiter=delimiter)
        fieldnames = reader.fieldnames
        return list(reader), fieldnames

def write_csv_from_dict(filename, rows, header_fields=None):
    with open(filename, 'w', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_fields)
        writer.writeheader()
        writer.writerows(rows)

def normalize_data(texts, remove_stopwords=True, remove_delex_slots=True):
    stopwords = nltk.corpus.stopwords.words('english')
    data = [w.lower() for w in nltk.word_tokenize(u" ".join(texts))
               if (w.lower() not in stopwords or not remove_stopwords) # Remove stopwords
               and ((remove_delex_slots and not w.startswith("__")) or not remove_delex_slots) # Remove delex slots
               and re.search(r'\w', w)]    # Remove words that do not have a word character.
    return data

def writeVocabFreqDist(output_fname, fd):
    list_of_dicts = []
    grams = fd.most_common(len(fd.keys()))
    count_dict = {}
    count_dicts = []
    for lexeme, count in grams:
        tmp_dict = {"token": lexeme, "frequency": count}
        list_of_dicts.append(tmp_dict)
        if count not in count_dict:
            count_dict[count] = 0
        count_dict[count] += 1
    for count, num_tokens in count_dict.items():
        tmp_dict = {"frequency":count, "num_unique_tokens":num_tokens}
        count_dicts.append(tmp_dict)
    count_dicts = sorted(count_dicts, key=lambda k: k['frequency'], reverse=True)

    headers = ["token", "frequency"]
    output_file = os.path.join("data", "emnlp", "freq_dist_tables_per_token", output_fname)
    write_csv_from_dict(output_file, list_of_dicts, headers)

    headers = ["frequency", "num_unique_tokens"]
    output_file = os.path.join("data", "emnlp", "freq_dist_tables_per_frequency", output_fname)
    write_csv_from_dict(output_file, count_dicts, headers)


def getVocabData(refs, num_grams=25, remove_stopwords=True, remove_delex_slots=True, get_vocab_data_freq_table=True, output_fname="vocab_data.csv"):
    normalized_refs = normalize_data(refs, remove_stopwords=remove_stopwords, remove_delex_slots=remove_delex_slots)

    uni_freq = nltk.FreqDist(normalized_refs)
    uni_fd = uni_freq
    print("=" * 70)
    print("Vocab size: " + str(len(uni_fd.keys())))

    print()
    print("Top %s Unigrams: "%str(num_grams))
    print(uni_fd.most_common(num_grams))

    ref_bigrams = nltk.bigrams(normalized_refs)
    bi_fd = nltk.FreqDist(u"{} {}".format(b1, b2) for b1, b2 in ref_bigrams)
    print()
    print("Top %s Bigrams: "%num_grams)
    print(bi_fd.most_common(num_grams))

    ref_trigrams = nltk.trigrams(normalized_refs)
    ti_fd = nltk.FreqDist(u"{} {} {}".format(t1, t2, t3) for t1, t2, t3 in ref_trigrams)
    print()
    print("Top %s Trigrams: "%num_grams)
    print(ti_fd.most_common(num_grams))

    ref_fourgrams = nltk.ngrams(normalized_refs, n=4)
    fi_fd = nltk.FreqDist(u"{} {} {} {}".format(q1, q2, q3, q4) for q1, q2, q3, q4 in ref_fourgrams)
    print()
    print("Top %s Fourgrams: " % num_grams)
    print(fi_fd.most_common(num_grams))

    if get_vocab_data_freq_table:
        writeVocabFreqDist(output_fname.split(".")[0]+"_unigrams.csv", uni_fd)
        writeVocabFreqDist(output_fname.split(".")[0]+"_bigrams.csv", bi_fd)
        writeVocabFreqDist(output_fname.split(".")[0]+"_trigrams.csv", ti_fd)
        writeVocabFreqDist(output_fname.split(".")[0]+"_fourgrams.csv", fi_fd)


def getUtteranceMetaData(refs):
    print("="*70)
    refs_word_tokenized = [len(nltk.word_tokenize(ref)) for ref in refs]
    mean = np.mean(refs_word_tokenized)
    print("Average # words/utterance: " + str(mean))

    refs_sent_tokenized = [len(nltk.sent_tokenize(ref)) for ref in refs]
    mean = np.mean(refs_sent_tokenized)
    print("Average # sentences/utterance: " + str(mean))

    sents = [nltk.sent_tokenize(ref) for ref in refs]
    sents = [val for sublist in sents for val in sublist]

    refs_word_tokenized = [len(nltk.word_tokenize(sent)) for sent in sents]
    mean = np.mean(refs_word_tokenized)
    print("Average # words/sentences: " + str(mean))

def getSlotCombinationDistribution():
    data_dir = os.path.join("data", "video_game", "individual_das")
    num_slots_dict = {}
    out_headers = ["mr_combo", "count"]
    for file in os.listdir(data_dir):
        rows, headers = read_csv(os.path.join(data_dir, file))
        if "errors" in file:
            continue
        match = re.search('video_games_da_(?P<da>.*?) \((?P<num_slots>\d) slots?\)', file)
        num_slots = int(match.groupdict()["num_slots"])
        da = match.groupdict()["da"]
        if num_slots not in num_slots_dict:
            num_slots_dict[num_slots] = {}
        if da not in num_slots_dict[num_slots]:
            num_slots_dict[num_slots][da] = {}
        for i, row in enumerate(rows):
            if i % 3 == 0:
                mr = row["mr"]
                video_game_slots = [video_game_slot for video_game_slot in VIDEO_GAME_SLOTS if video_game_slot+"[" in mr]
                video_game_slots.sort()
                video_game_slots_str = "_".join(video_game_slots)
                if video_game_slots_str not in num_slots_dict[num_slots][da]:
                    num_slots_dict[num_slots][da][video_game_slots_str] = 0
                num_slots_dict[num_slots][da][video_game_slots_str] += 1
    out_dir = os.path.join("data", "emnlp", "slot_combination_distribution_per_da")
    for num_slots_int, da_slots_dict in num_slots_dict.items():
        for da, slot_combo_dict in da_slots_dict.items():
            out_file = os.path.join(out_dir, da+"_"+str(num_slots_int)+"_combos.csv")
            out_dicts = []
            for combo, count in slot_combo_dict.items():
                out_dict = {"mr_combo":combo, "count":count}
                out_dicts.append(out_dict)
            out_dicts = sorted(out_dicts, key=lambda k: k['count'])
            write_csv_from_dict(out_file, out_dicts, header_fields=out_headers)

def getRefs(das=None, not_das=False, e2e=False, delex_data=False):
    refs = []
    if type(das) is str:
        das = [das]
    data_dir = os.path.join("data", "rest_e2e") if e2e else os.path.join("data", "video_game")
    vg_ds = ["test", "train", "valid"] if not delex_data else ["test [delex]", "train [delex]", "valid [delex]"]
    e2e_ds = ["trainset_e2e", "devset_e2e", "testset_e2e"] if not delex_data else ["trainset_e2e [delex]", "devset_e2e [delex]", "testset_e2e [delex]"]
    data_sets = e2e_ds if e2e else vg_ds
    for data_set in data_sets:
        rows, headers = read_csv(os.path.join(data_dir, data_set + ".csv"))
        for row in rows:
            da = row["mr"].split("(")[0]
            if das is None or (not not_das and da in das) or (not_das and da not in das):
                refs.append(row["ref"])
    return refs

def getMRs(das=None, not_das=False, data_set="train", slots=False, full_row=False, path="video_game"):
    refs = []
    if type(das) is str:
        das = [das]
    data_dir = os.path.join("data", path)
    if "e2e" in path:
        data_set += "set_e2e"
    rows, headers = read_csv(os.path.join(data_dir, data_set + ".csv"))
    for row in rows:
        da = row["mr"].split("(")[0]
        if das is None or (not not_das and da in das) or (not_das and da not in das):
            if slots:
                refs.append(row["mr"])
            elif full_row:
                refs.append((da, row))
            else:
                refs.append(da)
    return refs

def getDaDistribution():
    for data_set in ["train", "test", "valid"]:
        mr_dict = {}
        mrs = getMRs(data_set=data_set)
        for da in VIDEO_GAME_DAS:
            mr_dict[da] = mrs.count(da)
        sorted_x = sorted(mr_dict.items(), key=lambda kv: kv[1])
        # sorted_x = sorted((value, key) for (key, value) in mr_dict.items())
        print("="*70)
        print("This is the distribution for the %s dataset:"%data_set)
        for x,v in sorted_x:
            print(str(x)+": "+str(v))

def getSlotDistribution():
    for data_set in ["train", "test", "valid"]:
        for da in VIDEO_GAME_DAS:
            mr_dict = {}
            mrs = getMRs(das=[da], data_set=data_set, slots=True)
            all_video_game_slots = []
            for mr in mrs:
                video_game_slots = [video_game_slot for video_game_slot in VIDEO_GAME_SLOTS if video_game_slot + "[" in mr]
                all_video_game_slots += video_game_slots
            for video_game_slot in VIDEO_GAME_SLOTS:
                mr_dict[video_game_slot] = all_video_game_slots.count(video_game_slot)
            # sorted_x = sorted(mr_dict.items(), key=lambda kv: kv[1])
            sorted_x = sorted((key, value) for (key, value) in mr_dict.items())
            print("="*70)
            print("This is the distribution for the %s da in the %s dataset:"%(da, data_set))
            for x,v in sorted_x:
                print(str(x)+": "+str(v))

def generateHumanTrialsData(num_samples_all_but_inform=10, num_inform=40, data_set="test"):
    for path in ["rest_e2e", "video_game"]:
        mrs = getMRs(data_set=data_set, full_row=True, path=path)
        headers = ["mr", "ref"]
        output_rows = []
        if "e2e" in path:
            rows = [row for row_da, row in mrs]
            random.SystemRandom().shuffle(rows)
            output_rows += rows[:num_inform]
        else:
            for da in VIDEO_GAME_DAS:
                rows = [row for row_da, row in mrs if row_da == da]
                random.SystemRandom().shuffle(rows)
                min_val = num_samples_all_but_inform if da != "inform" else num_inform
                row_slice = min(min_val, len(rows))
                output_rows += rows[:row_slice]

        output_file = os.path.join("data", "emnlp", "human_trials_"+path+"_test.csv")
        write_csv_from_dict(output_file, output_rows, headers)

def main(combo_dist=True, vocab_data=True, utterance_meta_data=True, da_distribution=True, slot_distribution=True,
         get_vocab_data_freq_table=True, generate_human_trials_data=True):

    # get slot combo data from first mturk task
    if combo_dist:
        getSlotCombinationDistribution()

    # get vocab data
    if vocab_data:
        # These are the settings finalized in last meeting for use in paper
        remove_stopwords = False
        remove_delex_slots = True
        delex_data = True
        print("Video Game Data:")
        print("ALL DATA (remove_stopwords=%s):" % str(remove_stopwords))
        all_refs = getRefs(delex_data=delex_data)
        getVocabData(all_refs, remove_stopwords=remove_stopwords, remove_delex_slots=remove_delex_slots,
                     get_vocab_data_freq_table=get_vocab_data_freq_table, output_fname="all_data_remove_stopwords=%s.csv"%str(remove_stopwords))

        print("JUST INFORM DATA (remove_stopwords=%s):" % str(remove_stopwords))
        inform_refs = getRefs(["inform"], delex_data=delex_data)
        getVocabData(inform_refs, remove_stopwords=remove_stopwords, remove_delex_slots=remove_delex_slots,
                     get_vocab_data_freq_table=get_vocab_data_freq_table, output_fname="just_inform_remove_stopwords=%s.csv"%str(remove_stopwords))

        print("ALL BUT INFORM (remove_stopwords=%s):" % str(remove_stopwords))
        all_but_inform_refs = getRefs(["inform"], not_das=True, delex_data=delex_data)
        getVocabData(all_but_inform_refs, remove_stopwords=remove_stopwords, remove_delex_slots=remove_delex_slots,
                     get_vocab_data_freq_table=get_vocab_data_freq_table, output_fname="all_but_inform_remove_stopwords=%s.csv"%str(remove_stopwords))

        print("E2E Data (remove_stopwords=%s):" % str(remove_stopwords))
        e2e_refs = getRefs(e2e=True, delex_data=delex_data)
        getVocabData(e2e_refs, remove_stopwords=remove_stopwords, remove_delex_slots=remove_delex_slots,
                     get_vocab_data_freq_table=get_vocab_data_freq_table, output_fname="e2e_remove_stopwords=%s.csv"%str(remove_stopwords))

    # get utterance meta data
    if utterance_meta_data:
        all_refs = getRefs()
        data_dir = os.path.join("data", "video_game", "just_refs", "all.txt")
        with open(data_dir, mode="w") as file:
            for ref in all_refs:
                file.write(ref+"\n")
        getUtteranceMetaData(all_refs)
        e2e_refs = getRefs(e2e=True)
        data_dir = os.path.join("data", "video_game", "just_refs", "e2e.txt")
        with open(data_dir, mode="w") as file:
            for ref in e2e_refs:
                file.write(ref + "\n")
        getUtteranceMetaData(e2e_refs)
        for da in VIDEO_GAME_DAS:
            print("da: " + str(da))
            da_refs = getRefs(das=[da])
            getUtteranceMetaData(da_refs)
            data_dir = os.path.join("data", "video_game", "just_refs", da+".txt")
            with open(data_dir, mode="w") as file:
                for ref in da_refs:
                    file.write(ref+"\n")

    # get DA distribution
    if da_distribution:
        getDaDistribution()

    # get slot per DA distribution
    if slot_distribution:
        getSlotDistribution()

    if generate_human_trials_data:
        generateHumanTrialsData()

if __name__ == '__main__':
    main(False, False, False, False, False, False, True)