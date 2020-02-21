"""
File: consts.py
---------------
Contains constants used across the entire pipeline.

"""

OUTFILE_BBOXES = 'bboxes.json'
OUTFILE_EMBEDS = 'embeddings.json'
OUTFILE_BLACK_FRAMES = 'black_frames.json'
OUTFILE_METADATA = 'metadata.json'
OUTFILE_GENDERS = 'genders.json'
OUTFILE_IDENTITIES = 'identities.json'
OUTFILE_CAPTIONS = 'captions.srt'
OUTDIR_CROPS = 'crops'
OUTDIR_MONTAGES = 'montages'

SCANNER_COMPONENT_OUTPUTS = [
    OUTFILE_BBOXES,
    OUTFILE_EMBEDS,
    OUTFILE_METADATA,
    OUTDIR_CROPS
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

