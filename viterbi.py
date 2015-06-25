import sys
import nltk
import math
import collections

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    knownwords = []
    # a dictionary with default integer values to store the number of times a word appears 
    count = collections.defaultdict(int)
    
    # for each sentence in wbrown, count the number of time it appears
    for sentence in wbrown:
	for word in sentence:
	    count[word] += 1

    # for each word in the count dictionary, if the count is larger than 5, add it to knownwords list
    for k in count:
        if count[k] > 5:
            knownwords.append(k)   

    return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    rare = []
    str = '_RARE_'
    
    # for each sentence in file, replace words not in knownwords list with '_RARE_'
    for sentence in brown:
	sen = []
	for word in sentence:
	    if word in knownwords:
		sen.append(word)
	    else:
	        sen.append(str)

	rare.append(sen)

    return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')

    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()

#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
    # set qvalues to be a default integer value dictionary that stores the trigram log probabilities of each trigram 
    qvalues = collections.defaultdict(int)
    # set bi_c to be a default integer value dictionary that stores the count of bigrams 
    bi_c = collections.defaultdict(int)

    # for each sentence in file, count its trigram and bigram numbers
    for sentence in tbrown:
	bi_tuples = tuple(nltk.bigrams(sentence))
	for word in bi_tuples:
	    bi_c[word] += 1

	tri_tuples = tuple(nltk.trigrams(sentence))
	for word in tri_tuples:
	    qvalues[word] += 1
    # add two start symbols to bigram dictionary for trigram computation
    bi_c[('*','*')] = len(tbrown)

    # compute trigram log probabilities based on the counts above
    for k,v in qvalues.items():
	qvalues[k] = math.log(float(v)/bi_c[(k[0],k[1])], 2)

    
    return qvalues

#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    # set evalues to be a default integer value dictionary that stores the emission probability of every (word, tag) pair
    evalues = collections.defaultdict(int)
    # set tag_c to be a default integer value dictionary that stores the number of tags appears in the file
    tag_c = collections.defaultdict(int)
    # stores all existing tags
    taglist = []

    # for each sentence indexed at i in file
    for i in range(len(tbrown)):
	# for each word indexed at j in sentence i
	for j in range(len(tbrown[i])):
	    # update tag count and (word, tag) pair count
	    tag_c[tbrown[i][j]] += 1
	    evalues[(wbrown[i][j], tbrown[i][j])] += 1
    # for each entry in evalues dictionary, calculate (word, tag) probability by dividing corresponding tag counts
    for k,v in evalues.items():
	evalues[k] = math.log(float(v)/tag_c[k[1]], 2)

    # record all existing tags to taglist
    for k,v in tag_c.items():
	taglist.append(k)
    taglist.remove('*')
    taglist.remove('STOP')	

    return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords),
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a
#sentence tagged in the WORD/TAG format
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues: is from the return of calc_trigrams
#evalues is from the return of calc_emissions()
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    tagged = []

# for each sentence in brown, find its most possible tag sequence
    for sen in brown:
        # viberti value
        v = {}
        # tag path
        path = {}
        # initialize viterbi
        v[(0, '*', '*')] = 1

        # change rare words into '_RARE_' tokens
        sentence = []
        for word in sen:
            if word in knownwords:
                sentence.append(word)
            else:
                sentence.append('_RARE_')

        # for each word in the sentence, find its best tag
        for i in range(1, len(sentence) - 2):
            # for all possible tags tag1
            for tag1 in possible_tags(i, 2, taglist):
                # for all possible tags tag2
                for tag2 in possible_tags(i, 3, taglist):
                    # store viterbi for all possible tags tag3 
                    new_v = []
                    for tag3 in possible_tags(i, 1, taglist):
                        new_v.append( v[(i-1, tag3, tag1)] + get_value((tag3, tag1, tag2), qvalues) + get_value((sentence[i+1], tag2), evalues))

                    # find the max value among the new_v's, and the tag associated with this value
                    max_v = max(new_v)
                    maxarg_tag = possible_tags(i, 1, taglist)[new_v.index(max_v)]
                    # record this value to dictionary v and path
                    v[(i, tag1, tag2)] = max_v
                    path[(i, tag1, tag2)] = maxarg_tag

	# add the last triple (tag1, tag2, 'STOP')
	for tag1 in taglist:
	    for tag2 in taglist:
		v[(len(sentence) - 2, tag1, tag2)] = v[(len(sentence) - 3, tag1, tag2)] + get_value((tag1, tag2, 'STOP'), qvalues)

        # compute the highest probability for the sentence and the associated tags
        max_total = -10000000000
        # store the best two final tags 
        last_tag = ''
        second_last_tag = ''
        for tag1 in taglist:
            for tag2 in taglist:
                if v[(len(sentence) - 2, tag1, tag2)] > max_total:
                    max_total = v[(len(sentence) - 2, tag1, tag2)]
                    second_last_tag = tag1
		    last_tag = tag2
	if last_tag == '':
	    last_tag = 'NOUN'
	if second_last_tag == '':
	    second_last_tag = 'NOUN'

	# complete the whole path from end to start
        tag_path = [second_last_tag, last_tag, 'STOP']
        for i in range(len(sentence) - 3, 0, -1):
            tag_path.insert(0, path[(i, tag_path[0], tag_path[1])])

        # write the result to a string and append to a list
        tagged_sen = ''
        for word, tag in zip(sen[2:-1], tag_path[2:-1]):
            tagged_sen += word + '/' + tag + ' '
        tagged.append(tagged_sen)
	
    return tagged


#this funtion returns -1000 if the key is not in the dictionary, and the dictionary value if otherwise
def get_value(key, dictionary):
    if key in dictionary:
	return float(dictionary[key])
    else:
	return -1000

#this function returns a set of tags a word can use
def possible_tags(k, position, taglist):
    if k == 1:
        if position == 1 or position == 2:
            return ['*']
        else:
            return taglist
    elif k == 2:
        if position == 1:
           return ['*']
        else:
           return taglist
    elif k >= 3:
	return taglist

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence + '\n')
    outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of tagged sentences. Each sentence is in the WORD/TAG format and is a string rather than a list of tokens.
def nltk_tagger(brown):
    tagged = []
        
    training = nltk.corpus.brown.tagged_sents(tagset='universal')
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)

    # for each sentence in brown, tag and write into list
    for sen in brown:
	tagged_sen = tagger.tag(sen)
	for i in range(len(tagged_sen)):
	    tagged_sen[i] = tagged_sen[i][0] + '/' + tagged_sen[i][1]
	tagged.append(tagged_sen)

    return tagged

def q6_output(tagged):
    outfile = open('B6.txt', 'w')
    for sentence in tagged:
        output = ' '.join(sentence[2:-1]) + '\n'
        outfile.write(output)
    outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only)
def split_wordtags(brown_train):
    wbrown = []
    tbrown = []

    # for each sentence split word and tag
    for sentence in brown_train:
        token = sentence.split()

        word = ['/'.join((t.split('/'))[:-1]) for t in token]
        tag = [(t.split('/'))[-1] for t in token]
	
        # add start and stop symbols
	word.append('STOP')
	tag.append('STOP')
	word.insert(0, '*')
	word.insert(0, '*')
        tag.insert(0, '*')
	tag.insert(0, '*')
	
        wbrown.append(word)
        tbrown.append(tag)

    return wbrown, tbrown


def main():
    #open Brown training data
    infile = open("Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()
   
    #split words and tags, and add start and stop symbols (question 1)
    wbrown, tbrown = split_wordtags(brown_train)
    
    #calculate trigram probabilities (question 2)
    qvalues = calc_trigrams(tbrown)

    #question 2 output
    q2_output(qvalues)
   
    #calculate list of words with count > 5 (question 3)
    knownwords = calc_known(wbrown)
    
    #get a version of wbrown with rare words replace with '_RARE_' (question 3)
    wbrown_rare = replace_rare(wbrown, knownwords)
   
    #question 3 output
    q3_output(wbrown_rare)

    #calculate emission probabilities (question 4)
    evalues, taglist = calc_emission(wbrown_rare, tbrown)

    #question 4 output
    q4_output(evalues)
    
    #delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare

    #open Brown development data (question 5)
    infile = open("Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    #format Brown development data here
    brown = []
    for sentence in brown_dev:
	sen = nltk.word_tokenize(sentence)
	sen.insert(0,'*')
	sen.insert(0,'*')
	sen.append('STOP')	
     	brown.append(sen)
    brown_dev = brown

    #replace rare words in brown_dev (question 5)
    brown_dev_rare = replace_rare(brown_dev, knownwords)
    
    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)

    #question 5 output
    q5_output(viterbi_tagged)

    #do nltk tagging here
    nltk_tagged = nltk_tagger(brown_dev)

    #question 6 output
    q6_output(nltk_tagged)
if __name__ == "__main__": main()
