import requests
import json
import logging
import time
import os
import sys
import pathlib

logger = logging.getLogger(__name__)
sh = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(sh)

# API_WAIT = 1
API_WAIT = 0.5


url = 'https://api-free.deepl.com/v2/translate'
headers =  { 'Authorization': 'DeepL-Auth-Key 2e9bc943-35bd-d6f8-5fcb-eaa047035a2f:fx' }

def translate_as_deepl(text):
    payload = {
        'text': text,
        'target_lang': 'JA'
    }
    try:
        r = requests.post(url, headers=headers, data=payload)        
        print(r.text)
        result = json.loads(r.text)
        ts = result.get('translations', [])        
        t = ""
        if(len(ts) >= 1):
            t = ts[0].get("text")
        return t
    except Exception as e:
        logging.error(f'e: {e}, {text}')


class GASAPIException(Exception):
    pass

def translate_as_gas(data):
    """
    gasを使って翻訳

    data = [
        {
        instruction: '',
        input: '',
        output: '',
        }
    ]

    gasはここ
    https://script.google.com/home/projects/1uUMCKRb9aAo9SNHGd_AZPaU1oZlDKR2_Ikd4Oe2y_j0gOMlbDNfKtIU8/edit    
    """

    url = 'https://script.google.com/macros/s/AKfycbx9tWEViy4NNErCUHXIrphE7mzIr8rM5Ehnx3w1_Jo0b2D_DxTb5onlIYOOfimpMUOn/exec'
    headers = {
         "mode"       : "no-cors",
         "Content-Type" : "application/x-www-form-urlencoded",
    }

    payload = {
        'data': json.dumps(data)
    }
        
    try: 
        ## gas側で翻訳ごとに1s waitするので、chunk_size * 1s * 3(ins, inp, out) = 200 * 1 * 3 = 600sは最低限待つ
        r = requests.post(url, headers=headers, data=payload, timeout=(650.0, 650.0))

        if r.text.startswith("<!DOCTYPE html>"):            
            raise GASAPIException('api error')
        result = json.loads(r.text) 
        data = result.get('data', 'empty')
        if data == 'empty':
            return []
        else:            
            return data
    except GASAPIException:
        logging.error(f'api error {r.text}')
        sys.exit(1)
    except Exception as e:
        logging.error(f'e: {e}')
        return []

def translate_data_by_deepl(data):
    """
    data = [{'instruction': '', 'input': '', 'output': ''}]
    形式のdictの各valueを日本語に翻訳
    """
    translated_data = []
    for i in range(len(data)):
        logger.info(f'{i}/{len(data)}')        
        d = data[i]
        ins = d.get('instruction')
        inp = d.get('input')
        out = d.get('output')
        ins_t = translate_as_deepl(ins)
        inp_t = translate_as_deepl(inp)
        out_t = translate_as_deepl(out)
        r = {
            'instruction': ins_t,
            'input': inp_t,
            'output': out_t
        }
        translated_data.append(r) 
    return translated_data

def translate_data_by_gas(data, chunk_size = 200):
    translated_data = []

    total_records = len(data)
    num_chunks = (total_records + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        print(f'{i} / {num_chunks} (total {total_records})')
        start = i * chunk_size
        end = min(start + chunk_size, total_records)
        chunk_data = data[start:end]        
        r = translate_as_gas(chunk_data)        
        translated_data += r
        
    return translated_data

def translate_files_to_ja(target_dir: str, save_dir) -> None:
    """
    target_dirにあるjsonファイルを読み込み、日本語翻訳し、{元のファイル名}_ja.jsonとして保存する
    """

    files = os.listdir(target_dir)
    files = [
        'alpaca_data_4.json',
        'alpaca_data_3.json',
        'alpaca_data_2.json',
        'alpaca_data_1.json',
    ]
    files = ['alpaca_data_4.json']
    for filename in files:
        if filename.endswith('.json'):            
            target_file = str(pathlib.Path(target_dir) / filename)
            with open(target_file) as f:
                logger.info(f"load... {target_file}")
                org_data = json.load(f)
            
            # translated_data = translate_data_by_deepl(org_data)
            translated_data = translate_data_by_gas(org_data)

            basename, ext = os.path.splitext(filename)
            filename = f"{basename}_ja{ext}"
            save_file = str(pathlib.Path(save_dir) / filename)
            logger.info(f"save... {save_file}")
            with open(save_file, 'w') as f:
                json.dump(translated_data, f, ensure_ascii=False)

def split_json_file(filename, save_dir, chunk_size):
    """
    ファイル名を指定し、chunk_sizeごとに分割し、save_dirに保存する。
    保存されるファイル名は、{元のファイル名}_{ファイルカウント}.json
    """

    save_dir = pathlib.Path(save_dir)

    with open(filename, 'r') as f:
        data = json.load(f)
    total_records = len(data)
    num_chunks = (total_records + chunk_size - 1) // chunk_size
    basename, ext = os.path.splitext(filename)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_records)
        chunk_data = data[start:end]        
        chunk_filename = f"{basename}_{i+1}{ext}"
        save_file = str(save_dir / chunk_filename)
        logging.info(f'save... {save_file}')
        with open(save_file, 'w') as f:
            json.dump(chunk_data, f, ensure_ascii=False)

def combine_json_files(directory, output_filename):
    """
    指定したdirにあるjsonファイルを結合し、output_filenameに出力
    """

    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data.extend(json.load(f))
    with open(output_filename, 'w') as f:
        json.dump(data, f)


def test():
    target_file = './tmp_data.json'
    with open(target_file) as f:
        logger.info(f"load... {target_file}")
        org_data = json.load(f)        
        r = translate_data_by_gas(org_data)

def main():         
    split_json_file('./alpaca_data.json', './tmp', 20)
    # translate_files_to_ja('./data_set_tmp', './data_set')

if __name__ == '__main__':
    main()

