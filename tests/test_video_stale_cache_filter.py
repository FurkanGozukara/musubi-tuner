from types import SimpleNamespace

from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_KANDINSKY6, VideoDataset


def _dataset(target_frames=(121,), frame_extraction="head"):
    return SimpleNamespace(target_frames=target_frames, frame_extraction=frame_extraction, cache_directory="cache_dir")


def _selector(resolution):
    return BucketSelector(resolution, True, False, ARCHITECTURE_KANDINSKY6)


def test_clips_cached_for_another_resolution_or_length_are_ignored():
    selector = _selector((640, 352))
    current_bucket = selector.get_bucket_resolution((1920, 1080))
    current = f"clip1_00000-121_{current_bucket[0]:04d}x{current_bucket[1]:04d}_k6.safetensors"
    old_resolution = "clip1_00000-121_0640x0640_k6.safetensors"  # cached at 864x480 earlier
    old_length = f"clip1_00000-049_{current_bucket[0]:04d}x{current_bucket[1]:04d}_k6.safetensors"
    kept = VideoDataset.drop_stale_video_caches(_dataset(), [current, old_resolution, old_length], selector)
    assert kept == [current]


def test_unchanged_settings_keep_every_clip_and_all_stale_keeps_everything():
    selector = _selector((864, 480))
    files = ["a_00000-121_0640x0640_k6.safetensors", "b_00000-121_0864x0480_k6.safetensors"]
    assert all(selector.get_bucket_resolution((640, 640)) == (640, 640) for _ in [0])
    assert VideoDataset.drop_stale_video_caches(_dataset(), files, selector) == files
    # caching was not re-run for 512x288: nothing matches, the old behavior (use everything) stays
    assert VideoDataset.drop_stale_video_caches(_dataset(), files, _selector((512, 288))) == files
    # "full" extraction caches whole clips of any length
    assert VideoDataset.drop_stale_video_caches(_dataset(frame_extraction="full"), files, selector) == files
