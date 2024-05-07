import requests
import time
import json
from config import *
from tqdm import tqdm
from bs4 import BeautifulSoup # type: ignore

def get_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
    else:
        return None
    return soup, soup.prettify()

def parse_loops(soup):
    # parse loops from the website use BeautifulSoup

    tags_parents = soup.find_all('div', class_='tag-wrapper')

    records = {}
    for tags_parent in tags_parents:
        record = {}
        name = None
        
        tags = tags_parent.find_all('a')
        # extract the text from the tags
        tags_text = [tag.get_text(strip=True) for tag in tags]
        bpm = tags_text[0].strip().split(' bpm')[0]
        genre = tags_text[1].strip().split(' Loops')[0]
        # size = tags_text[3].strip()
        key = tags_text[5].strip().split('Key : ')[1]

        description_ = tags_parent.find_next_sibling('div', class_='desc-wrapper')  # 使用find_next_sibling查找描述
        if description_:
            descs = description_.find_all('p')

            temp = [desc.get_text(strip=True) for desc in descs][0].split('Description : ')
            if len(temp) > 1:
                description = temp[1].strip()
            else:
                description = ""

        player_wrapper = tags_parent.find_next_sibling('div', class_='player-wrapper')  # 使用find_next_sibling查找播放器部分
        if player_wrapper:
            link = player_wrapper.find('div', class_='player-title-wrapper-mbl').find('a')['href'] if player_wrapper else None

        mp3_div = tags_parent.find_next_sibling('div', class_='player-wrapper')  # 使用find_next_sibling查找MP3链接
        if mp3_div:
            mp3_link = mp3_div['rel']
            name = mp3_link.split('/')[-1].split('.')[0]
        
        # get the record
        if name:
            record_parent_key = name
            record['bpm'] = bpm
            record['genre'] = genre
            record['key'] = key
            record['description'] = description
            record['url'] = link
            records[record_parent_key] = record
    print(records)
    return records

def run(category, url, output_file):
    soup, _ = get_url(url)
    result = {}
    records = parse_loops(soup)
    result[category] = records
    with open(output_file, 'a') as f:
        f.write(json.dumps(result, indent=4))

def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        start_category_index = checkpoint['category_index']
        start_page = checkpoint['page']
    except (FileNotFoundError, json.JSONDecodeError):
        # if the checkpoint file does not exist, start from the beginning
        start_category_index = 0
        start_page = 1
    return start_category_index, start_page

def main():
    checkpoint_file = checkpoint_path + 'ckpt.json'
    start_category_index, start_page = load_checkpoint(checkpoint_file=checkpoint_file)

    for category_index, category in enumerate(tqdm(sorted_categories[start_category_index:])):
        category_index += start_category_index
        for page in tqdm(range(start_page, category['max_page']+1), desc=f"Processing {category['name']}"):
            
            cid = category['cid']
            category_name = category['name']
            url = "https://www.looperman.com/loops?page={0}&cid={1}&dir=d".format(page, cid)
            output_file = raw_path + '{0}_page_{1}.json'.format(category['name'], page)
            run(category=category_name, url=url, output_file=output_file)

            # save checkpoint
            checkpoint = {'category_index': category_index, 'page': page}
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

            # time.sleep(1)

        # reset start_page
        start_page = 1


if __name__ == '__main__':
    main()
    