# TV News Ingest Pipeline

## Getting Started

* Requires Python 3.5 or up

* Clone this repository

* Install Python dependencies with `pip3 install -r requirements.txt`

* Run `git submodule init && git submodule update`

* Install Docker (make sure your installation comes with `dockerd`)

* Setup the docker container for the Scanner dependency by following the
  instructions at http://scanner.run/guide/quickstart.html#quickstart (download 
  the `docker-compose.yml` file to the same directory as this repo)

## Usage

* Make sure videos are in the subtree from where `docker-compose.yml` is
  contained (this is a necessary consequence of how the docker image is created).

* All tv-news-ingest-pipeline scripts should be contained in same 
  directory as `docker-compose.yml`

* Ensure you have the docker daemon running (you can specify the host if 
  different than default)

### Notes for use with TV-News Viewer

* All video files must use the following naming convention:
`CHANNEL_YYYYMMDD_hhmmss_SHOW.mp4`

For example, `FOXNEWS_20180513_193000_The_Kelly_File.mp4`

### Run on a Single Video

To run for just a single video with path `my_video.mp4` for example:

Run `python3 pipeline.py my_video.mp4 output_dir`.

This will produce the following output:

```
output_dir/
├── bboxes.json         # bounding boxes per face
├── black_frames.json   # black frame locations
├── embeddings.json     # FaceNet embeddings for each face
├── genders.json        # male/female gender per face
├── identities.json     # celebrity identity per face
├── metadata.json       # number of frames, fps, name, width, height
└── crops               # cropped images of each face
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

* `genders` (skips gender classification)

* `captions` (skips copying over captions to output)

If you wanted to skip extracting face crops, for instance, you would run 

`python3 pipeline.py batch_videos.txt outputdir --disable face_crops black_frames`.

In this case, identifying faces with AWS will still be attempted, but will skip 
after finding that there are no face crops available.

If you run the pipeline once with certain features disabled, and then want to add them back later, 
you can rerun the pipeline with the same output directory and it will attempt to redo only what 
is necessary for the missing outputs.


### Useful Options

In addition to the `--captions` and `--disable` options mentioned previously, 
there are several other useful options or flags, all of which can be listed with 
`python3 pipeline.py --help`.

* `-r, --resilient`: Leaves the docker container up instead of taking it down after running. Should specify if planning to run multiple times.

* `-i, --init-run`: Skips checks for existing or cached outputs. Should specify if this is the first time you are running the script on the set of videos.

* `-f, --force`: Forces recomputation of all outputs, and overwrites existing ones.

* `--host`: Specify the IP:port that your docker daemon is running on if different than the default `127.0.0.1:2375`.


### Configuration

Further options can be configured in the `config` file. A sample is provided in 
this repository.

