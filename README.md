# TV News Ingest Pipeline

The TV News Ingest Pipeline is series of scripts designed to extract data and metadata from videos (specifically broadcast news). Though pipeline is intended for use with the [TV News Viewer](https://github.com/scanner-research/tv-news-viewer), where this module extracts (and formats) the data for the viewer to load, it can be used alone and produces human-readable information from the videos it processes. If you intend to use the pipeline outputs for TV News Viewer be sure to read the [TVNEWS_VIEWER_README.md](docs/TVNEWS_VIEWER_README.md) in the `docs` directory.

The usage of the pipeline will be explained further in the coming sections, but the idea is to take in a video file (or batch of video files) and seamlessly run a series of operations on the videos like detecting faces, identifying celebrities, classifying genders, and more. At the end of the pipeline, there will be outputs for each video in the specified output directory in an easy to use structure and format.


## Table of Contents

   * [Getting Started](#getting-started)
   * [Usage](#usage)
      * [Run on a Single Video](#run-on-a-single-video)
      * [Run on a Batch of Videos](#run-on-a-batch-of-videos)
      * [Run an Individual Script](#run-an-individual-script)
      * [Using Captions](#using-captions)
      * [Disabling Features](#disabling-features)
      * [Useful Options](#useful-options)
   * [Configuration](#configuration)
   * [Components](#components)
      * [Scanner Component](#scanner-component)
   
   
## Getting Started

1. Install Python3 (requires Python 3.5 or up)

2. Install Docker (if installing on Linux, make sure your installation comes with `dockerd`)

3. Clone this repository. The following instructions all take place within this repo.

4. Install Python dependencies with `pip3 install -r requirements.txt`

5. Clone submodule dependencies with `git submodule init && git submodule update`

6. Setup the docker container for the Scanner dependency by following the
  instructions at http://scanner.run/guide/quickstart.html#quickstart. You actually just need to run the following two commands:
   ```
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose pull cpu
   ```
   The first command downloads a `docker-compose.yml` file, and the second command downloads a ~5GB
   Docker container for Scanner, so it might take some time to complete.

7. If you want to enable the celebrity face identification using AWS, you will need to setup an account with AWS, and add your credentials to a `config.yml` file (see [Configuration](#configuration)). Learn more at https://docs.aws.amazon.com/rekognition/latest/dg/setting-up.html.

## Usage

* Make sure videos are in the subtree from where `docker-compose.yml` is
  contained (this is a necessary consequence of how the docker image is created).

* All tv-news-ingest-pipeline scripts should be contained in same 
  directory as `docker-compose.yml`

* Ensure you have the docker daemon running (you can specify the host if 
  different than default)
  
* Make sure you have the necessary capabilities to run each component that is active (read the [Components](#components) section for more information).

### Run on a Single Video

To run for just a single video with path `my_video.mp4` for example:

Run `python3 pipeline.py my_video.mp4 output_dir`.

This will produce the following output:

```
output_dir/
├── bboxes.json                 # bounding boxes per face
├── black_frames.json           # black frame locations
├── embeddings.json             # FaceNet embeddings for each face
├── genders.json                # male/female gender per face
├── identities.json             # celebrity identity per face
├── identities_propogated.json  # identities propogated to unlabeled faces
├── metadata.json               # number of frames, fps, name, width, height
└── crops                       # cropped images of each face
    ├── 0.png
    └── ...

```

### Run on a Batch of Videos

The more common usage of the pipeline, however, is to process videos in larger 
batches. In order to do so, the scripts take as input a text file of video 
file paths. For example:

`batch_videos.txt`:
```
path/to/my_video1.mp4
different/path/to/my_video2.mp4
```
Notably, all videos, whether specified in absolute or relative paths, must be 
contained in the subtree from where the `docker-compose.yml` file is located.

Then run `python3 pipeline.py batch_videos.txt output_dir`.

This will produce the following output:

```
output_dir/
├── my_video1
│   ├── bboxes.json        
│   ├── black_frames.json  
│   ├── crops              
│   │   ├── 0.png
│   │   └── ...
│   ├── embeddings.json    
│   ├── genders.json       
│   ├── identities.json
│   ├── identities_propogated.json
│   └── metadata.json  
└── my_video2
    ├── bboxes.json
    ├── black_frames.json
    ├── crops
    │   ├── 0.png
    │   └── ...
    ├── embeddings.json
    ├── genders.json
    ├── identities.json
    ├── identities_propogated.json
    └── metadata.json
```

### Run an Individual Script

If you want to run any of the pipeline components as individual scripts, use the 
`-s, --script` option. For instance, if you want to run just the scanner component, 

run `python3 pipeline.py batch.txt output_dir --script=scanner_component`,

or if you want to run just the gender classification,

run `python3 pipeline.py batch.txt output_dir --script=genders`.

Note, however, that you are responsible for making sure the requisite inputs 
exist for the component you would like to run (`embeddings.json` for the gender 
classifier, for example).


### Using Captions

Captions can be specified with the `--captions` option either as a single path to the `.srt` file or as a 
batch text file just as with videos. Make sure that the filename format of 
the captions are `<video_name>.srt` in order to match the caption with the video.

Run `python3 pipeline.py my_video.mp4 --captions=my_video.srt output_dir`

or

`python3 pipeline.py batch_videos.txt --captions=batch_captions.txt output_dir`.


### Disabling Features

You might not want to run all pipeline components on all videos or all the time. 
In this case, components can be disabled with the `-d` or `--disable` options.

Currently, the components that can be disabled are:

* `face_detection` (within `scanner_component.py`)

* `face_embeddings` (within `scanner_component.py`)

* `face_crops` (within `scanner_component.py`)

* `scanner_component` (skips the entire scanner component)

* `black_frames` (skips black frame detection)

* `identities` (skips face identification with AWS)

* `identity_propogation` (skips propogating identities to similar unlabeled faces)

* `genders` (skips gender classification)

* `captions` (skips copying over captions to output)

If you wanted to skip extracting face crops and detecting black frames, for instance, you would run 

`python3 pipeline.py batch_videos.txt outputdir --disable face_crops black_frames`.

In this case, identifying faces with AWS will still be attempted, but will skip 
after finding that there are no face crops available.

If you run the pipeline once with certain features disabled, and then want to add them back later, 
you can rerun the pipeline with the same output directory and it will attempt to redo only what 
is necessary for the missing outputs.


### Useful Options

In addition to the options mentioned previously, 
there are several other useful options or flags, all of which can be listed with 
`python3 pipeline.py --help`.

* `-r, --resilient`: Leaves the docker container up instead of taking it down after running. Should specify if planning to run multiple times.

* `-i, --init-run`: Skips checks for existing or cached outputs. Should specify if this is the first time you are running the script on the set of videos.

* `-f, --force`: Forces recomputation of all outputs, and overwrites existing ones.

* `--host`: Specify the IP:port that your docker daemon is running on if different than the default `127.0.0.1:2375`.


## Configuration

Further options can be configured in the `config` file, including things like credentials for the AWS identification service.
A sample is provided in this repository. The current configuration options are:

* `num_pipelines`: the number of pipelines to launch Scanner with (this likely should be fewer than the number of cores on your machine, e.g., 1/2 or 1/4 of the number of cores).

* `stride`: the interval in seconds between frames in which you look for faces.

* `montage_width`: the number of columns in the face image montage to send to AWS.

* `montage_height`: the number of rows in the face image montage to send to AWS.

* `aws_access_key_id`: your Amazon Rekognition access key ID.

* `aws_secret_access_key`: your Amazon Rekognition secret access key.


## Components

### Scanner Component
The scanner component consists of face detection, computing FaceNet embeddings, and extracting face image crops from frames. These are grouped together to reduce the overhead of decoding video frames.

The **face detection** component outputs a list of boundings boxes, in `bboxes.json`, one for each face detected. It is simply a JSON list of face IDs paired with the frame number they were located in and the bounding box information. Here is an example of one element in the list (where `27` is the ID of the face; face IDs count up from 0 per video):
```
[27, {"frame_num": 5580, "bbox": {"y2": 0.6054157018661499, "x2": 0.28130635619163513, "y1": 0.3943322002887726, "score": 0.9999775886535645, "x1": 0.1300494372844696}}]
```

The **face embeddings** component outputs a list of FaceNet embeddings, in `embeddings.json`, one for each face detected. It too is a JSON list of face IDs paired with the embedding vector of that face. Here is an example of one element in the list (where 13 is the ID of the face. The embeddings are quite long, so it's been truncated here):
```
[13, [-0.061142485588788986, ... , -0.007883959449827671]]
```

The **face crops** component outputs one image file per face detected, these reside in the `crops` directory and are named `<face_id>.png`. This image is the crop defined by the bounding box of the face (dilated a bit to give more room space along the edges). This output can be quite large, so be sure to delete the `crops` folder afterward if you aren't planning on using the face crops. (They are required for face identification, but if you don't want to do that component either you should disable `face_crops`.

### Black Frame Detection
This component detects black frames in the video (currently just for use in commercial detection). It outputs a list of the frame numbers of detected black frames, in `black_frames.json`. For example:
```
[22205, 22206, 22207, 23105, ..., 101293, 101294, 101295]
```

### Face Identification
This component attempts to recognize known celebrity identities from the face crop images. We currently use Amazon's Rekognition service to do so. It outputs a list of celebrity identities, in `identities.json`, for the face images it was able to identify. It is a JSON list of face IDs paired with their guessed identity and a confidence score of the guess between 0 and 100. Here is an example of one element in that list:
```
[213, "George W. Bush", 100.0]
```

### Face Identity Propogation
This component attempts to further identify faces that were failed to be labeled by the original face identification by comparing how similar unlabeled faces are to those that were labeled and *propogating* those identities to the unidentified faces if some thresholds are satisfied. The output is the same format as for [Face Identification](#face-identification), with these new identities appended to the end, in `identities_propogated.json`.

### Gender Classification
This components attempts to classify binary gender from the computed FaceNet embeddings. It outputs a list of guessed genders, in `genders.json`, one for each face detected. It is a JSON list of face IDs paired with a guess of male or female and a confidence score between 0 and 1. Here is an example of one item from that list:
```
[357, "F", 1.0]
```

### Captions
Currently, this component simply copies over the captions file specified into the output directory, as `captions.srt`. We are currently working on getting transcript alignment and commercial detection ready for the pipeline, and it will be coming soon!
