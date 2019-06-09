'''
Adapted from Matthew Mayo, KDnuggets
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
'''

import os
import re
import subprocess
import sys

from gensim.corpora import WikiCorpus

from constants import CORPORA_PATH

WIKIDUMP_URL = 'https://dumps.wikimedia.org/enwiki/latest/'


def wikipedia_dump_path(index=None):
    if index is None:
        return os.path.join(CORPORA_PATH, 'wikipedia')
    else:
        return os.path.join(CORPORA_PATH, 'wikipedia-' + str(index))


def list_wikipedia_dumps():
    proc = subprocess.run(
        args=['curl', '--silent', WIKIDUMP_URL],
        capture_output=True,
        text=True,
    )
    files = {}
    for line in proc.stdout.strip().splitlines():
        match = re.fullmatch('<a href="([^"]*)">.*', line)
        if not match:
            continue
        filename = match.group(1)
        url = os.path.join(WIKIDUMP_URL, filename)
        files[filename] = url
    return files


def determine_dump_url(index=None):
    if index is None:
        filename_start = 'enwiki-latest-pages-articles.xml'
    else:
        filename_start = 'enwiki-latest-pages-articles' + str(index) + '.xml'
    filename_end = '.bz2'
    url = None
    dump = list_wikipedia_dumps()
    for filename in dump:
        if filename.startswith(filename_start) and filename.endswith(filename_end):
            url = dump[filename]
            return filename, url
    raise RuntimeError('Cannot determine dump file URL')


def download_dump(index=None):
    filename, url = determine_dump_url(index)
    filepath = os.path.join(CORPORA_PATH, filename)
    subprocess.run(args=['curl', '--output', filepath, url])
    return filepath


def extract_corpus(infile, outfile):
    print(' '.join([
        'Extracting Wikipedia corpus file ' + infile + '.',
        'This may take a couple minutes...',
    ]))
    with open(outfile, 'w') as output:
        wiki = WikiCorpus(infile)
        # "text" is actually each individual article
        for i, text in enumerate(wiki.get_texts()):
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
            if i > 0 and i % 10000 == 0:
                print('Processed ' + str(i) + ' articles so far.')
    print('Processing complete! Yippee!')


def download_corpus(index=None):
    corpus_path = wikipedia_dump_path(index)
    if os.path.exists(corpus_path):
        return
    dumpfile = download_dump(index)
    extract_corpus(dumpfile, corpus_path)
    os.remove(dumpfile)


def main():
    index = None
    if len(sys.argv) == 2:
        index = int(sys.argv[1])
    elif len(sys.argv) > 2:
        print('usage: ' + sys.argv[0] + ' [index]')
        exit(1)
    download_corpus(index)


if __name__ == '__main__':
    main()
