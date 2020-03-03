"""
File: consts.py
---------------
Contains constants used across the entire pipeline.

"""

FILE_BBOXES = 'bboxes.json'
FILE_EMBEDS = 'embeddings.json'
FILE_BLACK_FRAMES = 'black_frames.json'
FILE_COMMERCIALS = 'commercials.json'
FILE_METADATA = 'metadata.json'
FILE_GENDERS = 'genders.json'
FILE_IDENTITIES = 'identities.json'
FILE_IDENTITIES_PROP = 'identities_propogated.json'
FILE_CAPTIONS = 'captions.srt'
FILE_CAPTIONS_ORIG = 'captions_orig.srt'
FILE_ALIGNMENT_STATS = 'alignment_stats.json'
DIR_CROPS = 'crops'

SCANNER_COMPONENT_OUTPUTS = [
    FILE_BBOXES,
    FILE_EMBEDS,
    FILE_METADATA,
    DIR_CROPS
]

LOCAL_TOML = """
# Scanner config
# Copy this to ~/.scanner/config.toml

[storage]
type = "posix"
db_path = "/root/.scanner/db"
[network]
worker_port = "5002"
master = "localhost"
master_port = "5001"
"""
