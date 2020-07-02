# Loading up the TV news viewer

Assuming everything went well and the metadata has been extracted, we will now
load the data into a form that the TV news viewer can understand.

Note: the viewer assumes files are named with the following format
`CHANNEL_YYYYMMDD_hhmmss_SHOW.mp4`.

## Getting the viewer code

Clone the TV news viewer repository.
https://github.com/scanner-research/tv-news-viewer.git

Perform the following steps in the README:

1. Install Rust (see https://rustup.rs/)
2. Clone submodules: git submodule init && git submodule update
3. Run ./install_deps.sh to install the submodules
  - Specifically, if you look inside this script, we are installing the
    caption index(er) and a library for handling video interval intersections
4. `cd vgrid-widget` and run `./install.sh`. This will require `npm` and other
   `js` dependencies. Install them as needed. Once this succeeds, `cd ..` to
   return to the top level.
5. Install python dependencies: `pip3 install -r requirements.txt`

## Preparing the data

Run `prepare_files_for_viewer.py <in_dir> <out_dir>` with `<in_dir>` as the path
to the directory containing output from the metadata extraction. This will put a
number of files into the output directory `<out_dir>`. See the script for more
detail as to what these files are.

## Putting it all together

Navigate to the directory where you cloned the TV news viewer. Copy
the data directory created by `prepare_files_for_viewer.py` as `data` here.
Once that is done, run `./derive_data.py` in the viewer repo.

You will want a place where the videos can be served. The easiest way to get
started is to serve them from the same machine as static files for now.
Symlink a directory containing the videos to `static/videos` in the
viewer directory.

Now, take a look at `develop.py` in the viewer repository.

You will want to change video endpoint and the default index path lines:
```
DEFAULT_VIDEO_ENDPOINT = '/static/videos'
DEFAULT_INDEX_PATH = 'data/index'     # we saved the index under data
```

You also want to change some of the defaults in the call to
`build_app`. See *** comments.

```
  app = build_app(
      data_dir, index_dir, video_endpoint, 
      video_auth_endpoint,                    # This is only needed if you have a
                                              # private and public video source
      fallback_to_archive=True,               # link videos directly from the archive
                                              # *** Make this False
      static_bbox_endpoint=None,
      static_caption_endpoint=None,
      host=None,
      min_date=datetime(2010, 1, 1),
      max_date=datetime(2029, 12, 31),
      tz=timezone('US/Eastern'),
      person_whitelist_file=None,
      min_person_screen_time=60 * 60,                     # *** Make this 0 (3600s by default)
      min_person_autocomplete_screen_time=10 * 60 * 60,   # *** Make this 0 (36000s by default)
      hide_person_tags=True,
      default_aggregate_by='month',           # *** Make this day unless you have years of data
      default_text_window=0,
      default_is_commercial=Ternary.false,
      default_color_gender_bboxes=True,
      allow_sharing=True,
      data_version='dev',
      show_uptime=True)
```

Once that is done, run `./develop.py` and navigate to `http://localhost:8080`
in your browser. The app should now be serving.

Note: for a more sensible/permanent deployment take a look at `wsgi.py` and
`config.json`. You can use this to serve the application with `uwsgi`,
`gunicorn`, or any other wsgi server.
