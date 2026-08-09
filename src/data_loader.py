import zipfile
import urllib.request
import os


def load_movielens_small(data_dir='data/'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    zip_path = os.path.join(data_dir, 'ml-latest-small.zip')
    
    if not os.path.exists(zip_path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)


if __name__ =='__main__':
    load_movielens_small()