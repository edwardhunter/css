#!/usr/bin/env python

"""
@package css
@file css/cr_retrieve.py
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


#<a href="/congressional-record/2009/senate-section/page/S1473-1614">S1473-1614</a>
#/congressional-record/2009/senate-section/page/S1991-2035
#<a href="http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9065-2">prayer</a>
#http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9078-8

# Possible network errors to be handled.
# IOError: [Errno socket error] [Errno 60] Operation timed out
# IOError: [Errno 54] Connection reset by peer


BASE_URL = 'http://beta.congress.gov'

PARA_START_EXPRS = re.compile(r'^\s{2}\w+',re.MULTILINE)
SPEECH_EXPR = re.compile(r'^ {,3}[^\s]+.*\n', re.MULTILINE)
#SPEAKER_EXPR = re.compile(r' {,3}(Mr.|Ms.|Mrs.) ([A-Za-z]+)(of \w+)?\.')
SPEAKER_EXPR = re.compile(r' {,3}(Mr.|Ms.|Mrs.) ([A-Za-z]+)\.?( of \w+\.)?')

# These patterns signale the end or interruption in a speech.
IGNORE_PATTERNS = [
    r'\s+The PRESIDING OFFICER',
    r'\s+ There being no objection',
    r'\s+The ACTING PRESIDENT',
    r'\s+The VICE PRESIDENT',
    r'\s+The (assistant )?legislative clerk',
    r'\s+The resolution|\s+The preamble',
    r'\s+At the request of',
    r'\s+The following .*were',
    r'\s+The yeas and nays',
    r'\s+S\.|\s+SA \d+',
    r'\s+At the request of',
    r'\s+The following .*were',
    r'\s+The yeas and nays',
    r'\s+[^\.]+submitted the following',
    r'\s+The (bill )?clerk',
    r'\s+Thereupon, the Senate',
    r'\s+Pending:',
    r'\s+(Mr.|Ms.|Mrs.) [A-Za-z]+ thereupon',
    r'\s+The Senate met',
    r'\s+The article follows',
    r'\s+The letter follows',
    r'\s+A message from the .+ was',
    r'\s+At .+(a|p)\.m\., a message from the',
    r'\s+\w+ received by the Senate:',
    r'\s+\w+There being no objection',
    r'\s+.+addressed the .+\.',                 #Mrs. HUTCHISON addressed the chair.
]

# These patterns are non speech content to be removed.
SUB_PATTERNS =[
    r'\[+.+]+\n+',
    r'__+',
    r'^\n'
]

NO_TRIES = 10

IGNORE_EXPRS = [re.compile(x) for x in IGNORE_PATTERNS]
SUB_EXPRS = [re.compile(x) for x in SUB_PATTERNS]


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
            time.sleep(10)

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
                time.sleep(10)

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
                    time.sleep(10)

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
    @result: Article filtered of non speech lines and text.
    """
    result = ''
    current_speaker = None
    if verbose:
        print '-'*100
        print 'NO SPEAKER:'

    for x in SPEECH_EXPR.finditer(art):

        line = x.group(0)
        speaker_match = SPEAKER_EXPR.match(line)

        # Case 1: line identifies a new speaker. Keep line.
        if speaker_match:
            current_speaker = speaker_match.group(2)
            #line = SPEAKER_EXPR.sub('',line)
            if verbose:
                print '-'*100
                print 'SPEAKER: ' + repr(current_speaker)

        # Case 2: existing speaker.
        # If interruption, reset speaker and eliminate line.
        elif current_speaker:
            for y in IGNORE_EXPRS:
                if y.match(line):
                    current_speaker = None
                    if verbose:
                        print '-'*100
                        print 'NO SPEAKER:'
                    break

        # Case 3: No existing speaker, line invalid.
        else:
            pass

        # If line valid, remove non speech text and append line.
        if current_speaker:
            for z in SUB_EXPRS:
                line = z.sub('', line)
            result += line
        if verbose:
            print line

    return result


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

    return result


def make_fname(congress_no, type):
    """
    Create filename for CR download or speech files.
    @param congress_no: int congress number.
    @param type: the file type string.
    @return fname: the file name.
    """
    fname = 'congress_%i_%s.pk' % (congress_no, type)
    return fname


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
    p.add_option('-s','--show', action='store_true', dest='show',
                help='Dump speech data to console.')

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
        f = open(fname, 'w')
        pickle.dump(articles, f)
        f.close()

    processed_speeches = []
    if opts.process:
        if not articles:
            fname = make_fname(congress_no, 'download')
            f = open(fname)
            articles = pickle.load(f)
            f.close()
        for i in range(len(articles)):
            filtered_text = filter_lines(articles[i][4])
            speeches = segment_speakers(filtered_text)
            for speech in speeches:
                processed_speeches.append((articles[i][0], articles[i][1],
                        articles[i][2], articles[i][3], speech[0], speech[1]))
        fname = make_fname(congress_no, 'speeches')
        f = open(fname, 'w')
        pickle.dump(processed_speeches, f)
        f.close()

    if opts.show:
        if not processed_speeches:
            fname = make_fname(congress_no, 'speeches')
            processed_speeches = pickle.load(open(fname))
        for x in processed_speeches:
            print '-'*80
            print '%i %s %s %s\n%s' % (x[0], str(x[1]), x[2], x[4], x[3])
            print '-'*80
            print x[5]

