import requests
import lxml.html as lh
import wget
import re


params = {

    'root': 'http://www.jsbach.net/midi/',
    'page': 'midi_solo_cello.html',
    'data_dir': './data/melody/'
    
}




def get_elements(params):
  
    url = params['root'] + params['page']

    html_page = requests.get(url)
    print('[+] Request Sent')
    doc = lh.fromstring(html_page.content)
    tr_elements = doc.xpath('//tr')
    print('[+] Elements Received')
    print(type(tr_elements))
    return tr_elements


def collect(tr_elements):

    collected = []

    idx, jdx = 0, 0
    for tr in tr_elements:
        jdx += 1    
        print(f'[i] TR: {jdx}')
        print(f'[i] ELEMENT: {tr}')
        for t in tr:
            idx += 1
            name=t.text_content()
            print(f'[+] {idx}: {name} collected...')
            collected.append(name)
    print('[+] Elements Collected')
    return collected


def collect_midi(collected):
    
    collect_midi = []
    pattern = '.mid$'

    for f in collected:
        x = re.search(pattern, f)
        if x:
            print(f'[i] File: {f} retained...')
            collect_midi.append(f)
    print('[+] Elements Filtered')
    return collect_midi


def gen_links(collected_midi, params):

    collected_links = []

    for m in collected_midi:
        link = params['root'] + f'{m}'
        collected_links.append(link)
    print('[+] Files --> Transform --> Links: success')
    return collected_links


def download_links(collected_links, params):

    for link in collected_links:
        wget.download(link, out=params['data_dir'])
    print('[i] New Files Located at: ' + params['data_dir'])
    return None


def fetch(params):
    print(' ----* PART 1:\n')
    elements = get_elements(params)
    print(' ----* PART 2:\n')
    collected = collect(elements)
    print(' ----* PART 3:\n')
    collected_midi = collect_midi(collected)
    print(' ----* PART 4: \n')
    collected_links = gen_links(collected_midi, params)
    print(' ----* PART 5:\n')
    download_links(collected_links, params)
    print(' ----* FETCHING & SAVING FILES COMPLETE')
    return None

if __name__ == "__main__":

    fetch(params)
