#!/usr/bin/env python3
import argparse
import requests
import os
import json
import time
import shutil
import glob
import random
import boto3
from tqdm import tqdm

AZURE_API_KEY = '0ca40107b225494ea7f2097c16aa10ee'
AZURE_RESOURCE = 'https://faces123.cognitiveservices.azure.com/'
CREDENTIAL_FILE = 'aws-credentials.csv'
CLIENT = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str)
    parser.add_argument('--limit', type=int)
    return parser.parse_args()

def read_img(img):
    with open(img, 'rb') as f:
        return f.read()

def load_client(credential_file):
    with open(credential_file) as f:
        f.readline()
        _, _, key_id, key_secret, _ = f.readline().split(',')

    return boto3.client(
        'rekognition', aws_access_key_id=key_id,
        aws_secret_access_key=key_secret,
        region_name='us-west-1')


def search_aws(img_file):
    img_data = read_img(img_file)
    global CLIENT
    if CLIENT is None:
        CLIENT = load_client(CREDENTIAL_FILE)
    # Supported image formats: JPEG, PNG, GIF, BMP.
    # Image dimensions must be at least 50 x 50.
    # Image file size must be less than 5MB.
    assert len(img_data) < 5e6, 'File too large: {}'.format(len(img_data))
    for i in range(1):
        try:
            resp = CLIENT.recognize_celebrities(Image={'Bytes': img_data})
            print(resp)
            return [x['Name'] for x in resp['CelebrityFaces']]
        except Exception as e:
            return ['Error']

def search_azure(img_file):
    # Supported image formats: JPEG, PNG, GIF, BMP.
    # Image dimensions must be at least 50 x 50.
    # Image file size must be less than 4MB.
    # Docs: https://westus.dev.cognitive.microsoft.com/docs/services/56f91f2d778daf23d8ec6739/operations/56f91f2e778daf14a499e1fa
    
    # From docs: The algorithm allows more than one face to be identified
    # independently at the same request, but no more than 10 faces.
    img_data = read_img(img_file)
    resp = requests.post(
        AZURE_RESOURCE + 'vision/v1.0/analyze',
        params={
            'details': 'Celebrities',
            'detectionModel': 'detect_02',
            'recognitionModel': 'recognition_02',
            'maxNumOfCandidatesReturned': 100
        },
        data=img_data,
        headers={
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': AZURE_API_KEY
        })
    names = {}
    if resp.status_code == 200:
        resp_json = resp.json()
        for x in resp_json['categories']:
            if ('people' in x['name'] and 'detail' in x
                    and 'celebrities' in x['detail']):
                for y in x['detail']['celebrities']:
                    name = y['name']
                    coordinates = y['faceRectangle']
                    xmean = coordinates['left'] + coordinates['width'] / 2
                    ymean = coordinates['top'] + coordinates['height'] / 2
                    names[int(xmean // 200), int(ymean // 200)] = y['name']
        # TODO: use confidence score
    else:
        print('Request failed: {}'.format(resp.status_code))
        return ['Error']
    return names

def load_json(fname):
    with open(fname) as f:
        return json.load(f)

def main(video_dir, limit):
    cluster_file = os.path.join(video_dir, 'clusters.json')
    clusters = load_json(cluster_file)

    img_files = [f for f in os.listdir(video_dir) if f.endswith('.png')]
    if limit:
        img_files = img_files[:limit]

    label_file = os.path.join(video_dir, 'azure_labels.json')
    if os.path.exists(label_file):
        face_id_to_names = load_json(label_file)
    else:
        face_id_to_names = {}
        for img_file in tqdm(img_files):
            face_id = img_file.split('.', 1)[0]
            img_path = os.path.join(video_dir, img_file)
            try:
                face_id_to_names[face_id] = search_azure(img_path)
            except Exception as e:
                print(e)
            time.sleep(3)
        with open(label_file, 'w') as f:
            json.dump(face_id_to_names, f)

    for i, cluster in enumerate(clusters):
        cluster_names = set()
        sample_ids = []
        for face_id in cluster:
            if str(face_id) in face_id_to_names:
                cluster_names.update(face_id_to_names[str(face_id)])
                sample_ids.append(face_id)
        sample_ids.sort()
        print('Cluster', i)
        print('  size:', len(cluster))
        print('  pred names:', ' , '.join(sorted(cluster_names)))
        print('  sample ids:', sample_ids)

    labeled_img_dir = os.path.join(video_dir, 'azure_labeled')
    if not os.path.exists(labeled_img_dir):
        os.makedirs(labeled_img_dir)

    for face_id in face_id_to_names:
        img_file = os.path.join(video_dir, face_id + '.png')
        for i, cluster in enumerate(clusters):
            if int(face_id) in cluster:
                cluster_id = i
                break
        if len(face_id_to_names[face_id]) > 0:
            names = '+'.join(set(face_id_to_names[face_id]))
            shutil.copyfile(
                img_file, os.path.join(
                    labeled_img_dir, '{:03d}.{}.{}.png'.format(
                        cluster_id, names, face_id)))
        else:
            shutil.copyfile(
                img_file, os.path.join(
                    labeled_img_dir, '{:03d}.unknown.{}.png'.format(
                        cluster_id, face_id)))

if __name__ == '__main__':
    # main(**vars(get_args()))
    print("globbing")
    files = list(sorted(glob.glob('02x02/*')))
    # random.shuffle(files)
    print("globbed", len(files))
    print("File\tAzure\tAmazon")
    for file in files[:5]: # files[:1000]:
        try:
            print("}\t{".join([file, str(search_azure(file)), ",".join(search_aws(file))]))
        except Exception as e:
            print("Failure", file, e)
