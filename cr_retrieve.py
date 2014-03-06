#!/usr/bin/env python

"""
@package css
@file css/cr_retrieve.py
@author Edward Hunter
@brief Utility module for retrieving congress.gov data.
"""

import urllib
import re

from bs4 import BeautifulSoup

opener = urllib.FancyURLopener({})


cr111_url = 'http://beta.congress.gov/congressional-record/111th-congress/browse-by-date'
pattern1 = '/congressional-record/\d{4}/senate-section/page/S.+'
expr1 = re.compile(pattern1)

#<a href="/congressional-record/2009/senate-section/page/S1473-1614">S1473-1614</a>
#/congressional-record/2009/senate-section/page/S1991-2035
f = opener.open(cr111_url)
soup1 = BeautifulSoup(f)
tags1 = soup1.find_all(href=expr1)

for tag in tags1:
    print tag['href']


cr111_s9056 = 'http://beta.congress.gov/congressional-record/2009/senate-section/page/S9065-9097'
pattern2 = '/congressional-record/\d{4}/\d{2}/\d{2}/senate-section/article/S.+'
expr2 = re.compile(pattern2)

#<a href="http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9065-2">prayer</a>
#http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9078-8
f = opener.open(cr111_s9056)
soup2 = BeautifulSoup(f)
tags2 = soup2.find_all(href=expr2)

for tag in tags2:
    print tag['href']


cr111_s9067_1 = 'http://beta.congress.gov/congressional-record/2009/08/07/senate-section/article/S9067-1'

f = opener.open(cr111_s9067_1)
soup3 = BeautifulSoup(f)
tags3 = soup3.find_all('pre')

for tag in tags3:
    print tag

