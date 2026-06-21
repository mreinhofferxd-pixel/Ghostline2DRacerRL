"""Track loading boundary: JSON on disk -> validated ``core.track.Track`` objects.

Kept out of ``core/`` on purpose: this is the I/O/validation adapter,
so the simulation itself never touches the filesystem. Dependency direction is
outer -> inner: this imports from ``core``, never the reverse.
"""

from .loader import TrackError, load_track, load_track_by_id, tracks_dir

__all__ = ["TrackError", "load_track", "load_track_by_id", "tracks_dir"]
