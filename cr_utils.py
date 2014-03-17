#!/usr/bin/env python

"""
@package css
@file css/cr_utils.py
@author Edward Hunter
@brief Utility module for retrieving congress.gov data.
"""

import urllib
import re
import time
import sys
import optparse
import pickle

from bs4 import BeautifulSoup

from data_utils import STATES, SENATORS

# CR daily and article tags and URLs.
#<a href="/congressional-record/2009/senate-section/page/S1473-1614">S1473-1614</a>
#/congressional-record/2009/senate-section/page/S1991-2035
#<a href="http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9065-2">prayer</a>
#http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9078-8

# DW Nominate Senate data URL.
DWN_URL = 'ftp://voteview.com/junkord/SL01112D21_BSSE.dat'

# DW Nominate line format and regex pattern.
# 112 29936 48 0 SOUTH ?? 200 DEMINT   ??  0.900  0.010  0.172  0.206  0.563   -59.22317  369   26  0.852
# regex group      1       2       3       4       5           6       7       8              9             10            11            12            13            14            15      16      17
DWN_PATTERN = r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([ A-Z]+)\s+(\d+)\s+([A-Z]+)([ A-Z\.]+)?\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+(\d+)\s+(\d+)\s+(-?[\d\.]+)'
DWN_EXPR = re.compile(DWN_PATTERN)

# DW Nominate fields.
"""
1 'congress',
2 'icpsr',
3 'state_code',
4 'district',
5 'state_name',
6 'party',
7 'name',
8 'coord_1',
9 'coord_2',
10 'coord_1_error',
11 'coord_2_error',
12 'corr',
13 'log_likelihood',
14 'no_votes',
15 'no_errors',
16'geom_mean_prob',
"""

DW_NOMINATE_FNAME = 'dw_nominate.txt'

# Possible network errors to be handled.
# IOError: [Errno socket error] [Errno 60] Operation timed out
# IOError: [Errno 54] Connection reset by peer

"""
Load combined files.
import pickle
x = pickle.loads(open('congress_112_combined.pkz','rb').read().decode('zip'))
"""

BASE_URL = 'http://beta.congress.gov'
SPEAKER_EXPR = re.compile(r' {,3}(Mr.|Ms.|Mrs.) ([A-Z][A-Za-z]+[A-Z])( of [^\.]+)?\.')

# These patterns signale the end or interruption in a speech.
STOP_PATTERNS = [
    # Observed in congress 112, 111, 110, 109
    r'\s+The (ACTING |VICE )?PRESIDENT',
    r'\s+The PRESIDING OFFICER',
    r'\s+The (articles?|letters?) follows?',
    r'\s+The (resolution|preamble)',
    r'\s+At the request of',
    r'\s+The following .*were',
    r'\s+The yeas and nays',
    r'\s+(Mr.|Ms.|Mrs.) [A-Za-z]+ (thereupon|addressed the)',
    r'\s+By (Mr.|Ms.|Mrs.) [A-Za-z]+.*',
    r'\s+S\.|\s+SA \d+',
    r'\s+The (bills?|results?|amendments?) (was|were)',
    r'\s+There being no objection',
    r'\s+The (assistant )?(legislative|bill) clerk',
    r'\s+Thereupon, the Senate',
    r'\s+Exhibit [A-Za-z0-9]+',
    # The following may be redundant or unnecessary.
    r'\s+[^\.]+submitted the following',
    r'\s+Pending:',
    r'\s+The Senate met',
    r'\s+A message from the .+ was',
    r'\s+At .+(a|p)\.m\., a message from the',
    r'\s+\w+ received by the Senate:',
    r'\s+\w+There being no objection',
]

# These patterns are non speech content to be removed.
REMOVE_PATTERNS =[
    r' *\[.+\]',
    r' *__+',
    r' {10,}[^ ].*'
]

# Network error retries and sleep time.
NO_TRIES = 10
SLEEP_TIME = 10

STOP_EXPRS = [re.compile(X) for X in STOP_PATTERNS]
REMOVE_EXPRS = [re.compile(X) for X in REMOVE_PATTERNS]


def download_articles_by_congress(congress_no, limit=None):
    """
    Download all senatorial articles from the congressional record for
    the given congress.
    @param congress_no: int between 104 and 113
    @param limit: int max number of articles to download.
    @return articles: list of text articles.
    """

    # Pattern for daily transcript senate tags.
    #<a href="/congressional-record/2009/senate-section/page/S1473-1614">S1473-1614</a>
    pattern1 = '/congressional-record/\d{4}/senate-section/page/S.+'
    expr1 = re.compile(pattern1)

    # Pattern for senate article tags on a particular day.
    #<a href="http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9065-2">prayer</a>
    pattern2 = '/congressional-record/(\d{4})/(\d{2})/(\d{2})/senate-section/article/S.+'
    expr2 = re.compile(pattern2)

    # Create the URL access.
    opener = urllib.FancyURLopener({})

    # Data structure to hold the articles.
    articles = []

    # Construct the top level url to the transcripts for the specified congress.
    cr_congress_url = '%s/congressional-record/%ith-congress/browse-by-date' % \
                      (BASE_URL, congress_no)

    # Download and parse the congress HTML and grab all senate tags.
    tries = 0
    while True:
        try:
            f = opener.open(cr_congress_url)
            soup1 = BeautifulSoup(f)
            tags1 = soup1.find_all(href=expr1)
            tries = 0
            break

        except IOError as ex:
            print 'ERROR: ' + str(ex)
            tries += 1
            if tries > NO_TRIES:
                raise
            time.sleep(SLEEP_TIME)

    # For each senate tag in the congress.
    starttime = time.time()
    count = 0
    #for tag1 in tags1[0:3]:
    for tag1 in tags1:
        count += 1
        # Construct the daily congress url.
        cr_daily_url = BASE_URL + tag1['href']
        print '-'*80
        print 'Extracting: %s' % cr_daily_url

        # Download and parse the daily transcript HTML.
        tries = 0
        while True:
            try:
                f = opener.open(cr_daily_url)
                soup2 = BeautifulSoup(f)
                tags2 = soup2.find_all(href=expr2)
                tries = 0
                break

            except IOError as ex:
                print 'ERROR: ' + str(ex)
                tries += 1
                if tries > NO_TRIES:
                    raise
                time.sleep(SLEEP_TIME)

        # For each article tag in the daily transcript.
        for tag2 in tags2:
            if limit and len(articles) >= limit:
                return articles

            cr_article_url =  tag2['href']
            # Get date (year, month, day)
            rhs = cr_article_url.replace(BASE_URL, '')
            m = expr2.match(rhs)
            date = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            # Get article ID.
            article_id = cr_article_url.split('/')[-1]
            article_title = tag2.get_text()

            print '...Extracting: %s' % cr_article_url
            tries = 0
            while True:
                try:
                    f = opener.open(cr_article_url)
                    soup3 = BeautifulSoup(f)
                    tags3 = soup3.find_all('pre')
                    tries = 0
                    break

                except IOError as ex:
                    print 'ERROR: ' + str(ex)
                    tries += 1
                    if tries > NO_TRIES:
                        raise
                    time.sleep(SLEEP_TIME)

            # Extract article text.
            if len(tags3)!=1:
                print '-'*80
                print 'WARNING: multiple texts found!'
                print '-'*80
            for tag3 in tags3:
                articles.append((congress_no, date, article_id, article_title,
                                 tag3.get_text()))

        print 'Retrieved %i articles.' % len(articles)
        delta = (time.time() - starttime)/60.0
        print 'Processed %f percent of Congress %i dailies in %f minutes.' % \
              (float(count)/float(len(tags1)), congress_no, delta)
    return articles


def filter_lines(art, verbose=False):
    """
    Remove non speech lines and text.
    @param art: Article to filter.
    @param verbose: Boolean to output filtered text during run for debugging.
    @return (result, patterns): Tuple of article text filtered for speech
    text only, set of unique regex patterns used in detecting speech endings.
    """
    result = ''
    patterns = set()
    current_speaker = None
    if verbose:
        print '-'*80
        print 'BEGIN:'

    lines = art.split('\n')
    for i, line in enumerate(lines):
        speaker_match = SPEAKER_EXPR.match(line)
        if speaker_match:
            current_speaker = speaker_match.group(2)
            if verbose:
                print '-'*80
                print 'SPEAKER: ' + repr(current_speaker)
                print line

        elif current_speaker:
            for y in STOP_EXPRS:
                ymatch = y.match(line)
                if ymatch:
                    patterns.update([ymatch.re.pattern])
                    current_speaker = None
                    if verbose:
                        print '-'*80
                        print 'NO SPEAKER:'
                    break

        else:
            pass

        # If line valid, remove non speech text and append line.
        if current_speaker:
            for z in REMOVE_EXPRS:
                line = z.sub('', line)
            result += line
            if len(line)>0:
                result += '\n'

        if verbose:
            print line

    return result, patterns


def segment_speakers(art):
    """
    Segment speaker texts from articles.
    @param art: Senate article.
    @return result: list of (speaker, text).
    """

    result = []
    current_speaker = None
    lines = art.split('\n')
    speech = ''
    for line in lines:
        speaker_match = SPEAKER_EXPR.match(line)
        if speaker_match:
            if current_speaker:
                result.append((current_speaker, speech))
                speech = ''
            current_speaker = speaker_match.group(2)
            line = SPEAKER_EXPR.sub('',line)
            speech = line
        elif current_speaker:
            speech += '\n'
            speech += line

    if len(speech) >0 and current_speaker:
        speech = filter_states_and_senators(speech)
        result.append((current_speaker.upper(), speech))

    return result


def filter_states_and_senators(speech):
    """
    Filter out state and senator names.
    @param speech: Speech text to filter.
    @return speech: filtered speech.
    """

    senator_exprs = []
    for x in SENATORS:
        pattern = r'(Mr\.\s+|Mrs\.\s+|Ms\.\s+|Senator\s+)([A-Z]+\s+)?%s' % x
        senator_exprs.append(re.compile(pattern, re.MULTILINE|re.IGNORECASE))

    state_pattern1 = r''
    state_pattern2 = r''
    for x in STATES:
        x = x.replace(' ','\s+')
        state_pattern1 += r' %s|' % x
        state_pattern2 += r'\n%s|' % x
    state_pattern1 = state_pattern1[:-1]
    state_pattern2 = state_pattern2[:-1]
    state_exp1 = re.compile(state_pattern1, re.MULTILINE|re.IGNORECASE)
    state_exp2 = re.compile(state_pattern2, re.MULTILINE|re.IGNORECASE)

    for x in senator_exprs:
        speech = x.sub('SENATOR_NAME', speech)
    speech = state_exp1.sub(' STATE_NAME', speech)
    speech = state_exp2.sub('\nSTATE_NAME', speech)

    return speech


def combine_speeches(speeches):
    """
    Combine speeches from each speaker into a single document keyed
    by speaker name.
    @param sppeches. List of speech tuples.
    @return result: dict of combined speeches.
    """
    result = {}
    for speech in speeches:
        speaker = speech[4]
        if speaker not in result.keys():
            result[speaker] = ''
        result[speaker] += speech[5]
        result[speaker] += '\n\n'
    return result


def make_fname(congress_no, type, ext='pkz'):
    """
    Create filename for CR download or speech files.
    @param congress_no: int congress number.
    @param type: the file type string.
    @return fname: the file name.
    """
    fname = 'congress_%i_%s.%s' % (congress_no, type, ext)
    return fname


def create_dwn(fname=DW_NOMINATE_FNAME):
    """
    Parse and save a dw-nominate data object
    @param fname: File name for dw-nominate senate data.
    @return result: xxx.
    """

    # Read in dw nominate file.
    try:
        f = open(fname)
    except:
        # download and open dw nominate data.
        pass

    data = f.read()
    f.close()

    # Parse lines into a dict.
    scores_dict = {}
    lines = data.split('\n')
    for line in lines:
        m = DWN_EXPR.match(line)
        if m:
            congress            = int(m.group(1))
            icpsr               = int(m.group(2))
            state_code          = int(m.group(3))
            district            = int(m.group(4))
            state               = m.group(5).lstrip().rstrip()
            party               = int(m.group(6))
            name                = m.group(7).lstrip().rstrip()
            coord_1             = float(m.group(9))
            coord_2             = float(m.group(10))
            coord_1_error       = float(m.group(11))
            coord_2_error       = float(m.group(12))
            corr                = float(m.group(13))
            log_likelihood      = float(m.group(14))
            no_votes            = int(m.group(15))
            no_errors           = int(m.group(16))
            geom_mean_prob      = float(m.group(17))
            if state != 'USA':
                key = (congress, name)
                val = [congress, icpsr, state_code, district, state, party,
                       name, coord_1, coord_2, coord_1_error, coord_2_error,
                       corr, log_likelihood, no_votes, no_errors, geom_mean_prob]
                scores_dict[key] = val

    f = open('dw_nominate.pkz','wb')
    f.write(pickle.dumps(scores_dict).encode('zip'))
    f.close()
    senators = set([x[1] for x in scores_dict.keys() if x[0]>=103])
    f = open('senators.pkz','wb')
    f.write(pickle.dumps(senators).encode('zip'))
    f.close()


def create_labled_data(congress_no_train=107, congress_no_test=112, count=25,
                       fname=DW_NOMINATE_FNAME):
    """
    Create labled data for a given contress.
    @param congress_no: Int congress number to label.
    """

    # Load DW Nominate scores.
    try:
        f = open('dw_nominate.pkz','rb')
    except:
        create_dwn(fname)
        f = open('dw_nominate.pkz','rb')
    scores = pickle.loads(f.read().decode('zip'))
    f.close()
    vals = scores.values()

    # Assemble training data.
    train = []
    train_target = []
    for congress in range(congress_no_train,congress_no_test):
        fname = make_fname(congress,'combined')
        congress_data = pickle.loads(open(fname, 'rb').read().decode('zip'))

        conservative_scores = [(x[6], x[7])  for x in vals if x[0] == congress and x[7]>0]
        conservative_scores.sort(key= lambda tup: tup[1], reverse=True)
        conservative_scores = conservative_scores[:25]
        liberal_scores = [(x[6], abs(x[7]))  for x in vals if x[0] == congress and x[7]<0]
        liberal_scores.sort(key= lambda tup: tup[1], reverse=True)
        liberal_scores = liberal_scores[:25]

        for i in range(count):
            try:
                name = conservative_scores[i][0]
                train.append(congress_data[name])
                train_target.append(1)
            except KeyError:
                print 'ERROR no data for: %s in congress %i' % (name, congress)
            try:
                name = liberal_scores[i][0]
                train.append(congress_data[name])
                train_target.append(0)
            except KeyError:
                print 'ERROR no data for: %s in congress %i' % (name, congress)

    # Assemble test data.
    test = []
    test_target = []
    fname = make_fname(congress_no_test,'combined')
    congress_data = pickle.loads(open(fname, 'rb').read().decode('zip'))
    for i in range(count):
        conservative_scores = [(x[6], x[7])  for x in vals if x[0] == congress_no_test and x[7]>0]
        conservative_scores.sort(key= lambda tup: tup[1], reverse=True)
        conservative_scores = conservative_scores[:25]
        liberal_scores = [(x[6], abs(x[7]))  for x in vals if x[0] == congress_no_test and x[7]<0]
        liberal_scores.sort(key= lambda tup: tup[1], reverse=True)
        liberal_scores = liberal_scores[:25]

        try:
            name = conservative_scores[i][0]
            test.append(congress_data[name])
            test_target.append(1)
        except KeyError:
            print 'ERROR no data for: %s in congress %i' % (name, congress_no_test)

        try:
            name = liberal_scores[i][0]
            test.append(congress_data[liberal_scores[i][0]])
            test_target.append(0)
        except KeyError:
            print 'ERROR no data for: %s in congress %i' % (name, congress_no_test)

    # Create and save data object.
    data = {}
    data['target_names'] = ['liberal', 'conservative']
    data['train'] = train
    data['train_target'] = train_target
    data['test'] = test
    data['test_target'] = test_target
    f = open('senate.pkz', 'wb')
    f.write(pickle.dumps(data).encode('zip'))
    f.close()


def dump_states_and_senators():
    """
    Print a list of state names and senators to the console.
    """
    # Load DW Nominate scores.
    try:
        f = open('dw_nominate.pkz','rb')
    except:
        create_dwn(fname)
        f = open('dw_nominate.pkz','rb')
    scores = pickle.loads(f.read().decode('zip'))
    f.close()
    keys = scores.keys()
    vals = scores.values()

    senators = set([x[1] for x in keys])
    for x in senators:
        print x
    states = set([x[4] for x in vals])
    print '\n\n'
    print str(states)

if __name__ == '__main__':

    # Parse command line arguments and options.
    usage = 'usage: %prog [options] congress'
    usage += '\n congress = congress number in range 104-113.'
    description = 'Download and process congressional record.'
    p = optparse.OptionParser(usage=usage, description=description)
    p.add_option('-d','--download', action='store_true', dest='download',
                 help='Congress number to download and store.')
    p.add_option('-l','--limit', action='store', dest='limit',
                 type='int', help='Limit the number of article downloads.')
    p.add_option('-p','--process', action='store_true', dest='process',
                help='Process congress file for speech content.')
    p.add_option('-t','--text', action='store_true', dest='text',
                help='Create text file of processed congress data.')
    p.add_option('-v','--verbose', action='store_true', dest='verbose',
                help='Dump debugging data to console.')
    p.add_option('-e','--expressions', action='store_true', dest='expressions',
                help='Dump stop expression data to console.')

    (opts, args) = p.parse_args()
    if len(args) < 1:
        p.print_usage()
        sys.exit(1)

    congress_no = int(args[0])
    articles = None
    if opts.download:
        limit = opts.limit
        articles = download_articles_by_congress(congress_no, limit)
        fname = make_fname(congress_no, 'download')
        f = open(fname, 'wb')
        f.write(pickle.dumps(articles).encode('zip'))
        f.close()

    processed_speeches = []
    all_patterns = set()
    if opts.process:
        if not articles:
            fname = make_fname(congress_no, 'download')
            f = open(fname, 'rb')
            articles = pickle.loads(f.read().decode('zip'))
            f.close()

        for i in range(len(articles)):
            filtered_text, patterns = filter_lines(articles[i][4], opts.verbose)
            all_patterns.update(patterns)
            speeches = segment_speakers(filtered_text)
            for speech in speeches:
                processed_speeches.append((articles[i][0], articles[i][1],
                        articles[i][2], articles[i][3], speech[0], speech[1]))
            print 'Processing congress %s      %f complete.' % \
                  (congress_no, float(i)/float(len(articles)))
        speech_dict = combine_speeches(processed_speeches)

        fname = make_fname(congress_no, 'speeches')
        f = open(fname, 'wb')
        f.write(pickle.dumps(processed_speeches).encode('zip'))
        f.close()
        fname = make_fname(congress_no, 'combined')
        f = open(fname, 'wb')
        f.write(pickle.dumps(speech_dict).encode('zip'))
        f.close()

        if opts.verbose or opts.expressions:
            print '-'*80
            print 'STOP PATTERNS:'
            for x in all_patterns:
                print repr(x)

    if opts.text:
        if not processed_speeches:
            fname = make_fname(congress_no, 'speeches')
            f = open(fname, 'rb')
            articles = pickle.loads(f.read().decode('zip'))
            f.close()
        fname = make_fname(congress_no,'dump','txt')
        f = open(fname,'w')
        for x in processed_speeches:
            metadata = '\n\n' + '-'*120 + '\n'
            metadata += 'congress: %i date: %s article: %s speaker: %s' % \
                        (x[0], str(x[1]), x[2], x[4])
            metadata += '\ntitle: %s\n\n' % x[3]
            f.write(metadata)
            f.write(x[5])
        f.close()

