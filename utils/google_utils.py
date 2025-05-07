# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform  # 获取平台信息
import subprocess  # 子进程调用
import time
from pathlib import Path  # 路径操作

import requests  # HTTP请求库
import torch


def gsutil_getsize(url=''):
    # 使用gsutil命令获取Google Cloud Storage文件的大小 gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # 返回文件大小（单位：字节），如果输出为空则返回0


def attempt_download(file, repo='ultralytics/yolov5'):
    # 尝试下载文件（如果文件不存在）
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # 获取GitHub最新发布信息
            assets = [x['name'] for x in response['assets']]  # 提取发布中的文件名['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # i.e. 'v1.0'
        except:  # 如果请求失败，使用默认值
            assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v5.0'  # 默认版本

        name = file.name
        if name in assets:
            msg = f'{file} missing, try downloading from https://github.com/{repo}/releases/'
            redundant = False  # 是否有备用下载选项
            try:  # GitHub
                url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
                print(f'Downloading {url} to {file}...')
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1E6  # 检查文件是否存在且大小大于1MB
            except Exception as e:  # GCP
                print(f'Download error: {e}')
                assert redundant, 'No secondary mirror'
                url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
                print(f'Downloading {url} to {file}...')
                os.system(f'curl -L {url} -o {file}')  # 使用curl下载文件
            finally:
                if not file.exists() or file.stat().st_size < 1E6:  # check
                    file.unlink(missing_ok=True)  # 删除部分下载的文件
                    print(f'ERROR: Download failure: {msg}')
                print('')
                return


def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    # 从Google Drive下载文件 from yolov5.utils.google_utils import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')  # Google Drive cookie文件
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    file.unlink(missing_ok=True)  # 删除已存在的文件
    cookie.unlink(missing_ok=True)  # 删除已存在的cookie文件

    # 尝试下载文件
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # 执行下载命令
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # 删除部分下载的文件
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        os.system(f'unzip -q {file}')  # unzip
        file.unlink()  # 删除zip文件

    print(f'Done ({time.time() - t:.1f}s)')
    return r


def get_token(cookie="./cookie"):  # 从cookie文件中获取token
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
