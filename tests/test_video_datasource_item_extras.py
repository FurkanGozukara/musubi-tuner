import json

from musubi_tuner.dataset.image_video_dataset import VideoDirectoryDatasource, VideoJsonlDatasource


def test_video_directory_items_have_no_extras(tmp_path):
    # MiniMax H3 latent caching reads get_item_extras for every item; the fork's video datasources lacked it
    (tmp_path / "clip1.mp4").write_bytes(b"0")
    (tmp_path / "clip1.txt").write_text("a clip", encoding="utf-8")
    source = VideoDirectoryDatasource(str(tmp_path), caption_extension=".txt")
    extras = source.get_item_extras(0)
    assert extras.fields == {} and extras.base_directory == str(tmp_path) and extras.label.endswith("clip1.mp4")


def test_video_jsonl_extras_skip_the_shared_and_fork_keys(tmp_path):
    jsonl = tmp_path / "clips.jsonl"
    record = {
        "video_path": "a.mp4", "caption": "x", "audio_path": "a.wav", "loss_mask_path": "m.png",
        "short_caption": "y", "references": [{"path": "r.png"}],
    }
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    extras = VideoJsonlDatasource(str(jsonl), caption_field="short_caption").get_item_extras(0)
    assert set(extras.fields) == {"references"}
    assert extras.label == "clips.jsonl line 1"
