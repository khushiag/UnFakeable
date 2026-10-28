import pytest

from unfakeable.data.manifest import (assign_groups, build_celebdf_manifest, build_ffpp_manifest, inspect_datasets,
                                      subsample_ffpp)
from unfakeable.data.splits import LeakageError, assert_group_disjoint, assign_train_test


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


@pytest.fixture
def ffpp_tree(tmp_path):
    n_pairs = 12
    for i in range(2 * n_pairs):
        _touch(tmp_path / "original_sequences/youtube/c23/videos" / f"{i:03d}.mp4")
    for m in ("Deepfakes", "FaceSwap"):
        for k in range(n_pairs):
            for a, b in ((2 * k, 2 * k + 1), (2 * k + 1, 2 * k)):
                _touch(tmp_path / f"manipulated_sequences/{m}/c23/videos/{a:03d}_{b:03d}.mp4")
    return tmp_path


def test_groups_link_sources():
    g = assign_groups([["000"], ["001"], ["000", "001"], ["001", "000"], ["002"]], "x")
    assert g[0] == g[1] == g[2] == g[3] and g[4] != g[0]


def test_ffpp_manifest_labels_and_groups(ffpp_tree):
    df = build_ffpp_manifest(str(ffpp_tree), "c23", ["Deepfakes", "FaceSwap"])
    assert (df.label == 0).sum() == 24 and (df.label == 1).sum() == 48
    assert df.group.nunique() == 12  # one group per source pair
    fake = df[df.video_id == "ffpp__Deepfakes__000_001"].iloc[0]
    real = df[df.video_id == "ffpp__original__001"].iloc[0]
    assert fake.group == real.group  # a fake shares its group with BOTH source videos


def test_ffpp_manifest_tolerates_flat_layout_and_wrapper(tmp_path):
    root = tmp_path / "wrap" / "FaceForensics++"
    for i in range(4):
        _touch(root / "original_sequences/youtube/c23" / f"{i:03d}.mp4")  # no "videos" folder
    _touch(root / "manipulated_sequences/Deepfakes" / "000_001.mp4")  # flat method folder
    df = build_ffpp_manifest(str(tmp_path / "wrap"), "c23", ["Deepfakes"])
    assert (df.label == 0).sum() == 4 and (df.label == 1).sum() == 1
    assert any("original (real) :     4" in line for line in
               inspect_datasets(str(tmp_path / "wrap"), str(tmp_path), "c23", ["Deepfakes"]))


def test_split_is_group_disjoint_and_deterministic(ffpp_tree):
    df = build_ffpp_manifest(str(ffpp_tree), "c23", ["Deepfakes", "FaceSwap"])
    df = subsample_ffpp(df, max_real=24, max_fake=None, balance=True, seed=3)
    assert (df.label == 0).sum() == (df.label == 1).sum()
    s1, s2 = assign_train_test(df, 0.25, 3), assign_train_test(df, 0.25, 3)
    assert s1.equals(s2) and set(s1) == {"train", "test"}
    assert_group_disjoint(df.assign(split=s1))
    # every group (source pair) is entirely on one side
    assert df.assign(split=s1).groupby("group").split.nunique().max() == 1


def test_leak_detection_raises(ffpp_tree):
    df = build_ffpp_manifest(str(ffpp_tree), "c23", ["Deepfakes"])
    leaky = df.assign(split=["train", "test"] * (len(df) // 2))
    with pytest.raises(LeakageError):
        assert_group_disjoint(leaky)


def test_subsample_balances_methods(ffpp_tree):
    df = build_ffpp_manifest(str(ffpp_tree), "c23", ["Deepfakes", "FaceSwap"])
    sub = subsample_ffpp(df, 16, 16, True, 0)
    counts = sub[sub.label == 1].method.value_counts()
    assert counts.max() - counts.min() <= 1 and len(sub[sub.label == 0]) == 16


def test_celebdf_manifest_and_official_list(tmp_path):
    for rel in ("Celeb-real/id0_0000.mp4", "YouTube-real/00001.mp4", "Celeb-synthesis/id0_id1_0000.mp4",
                "Celeb-synthesis/id2_id3_0000.mp4"):
        _touch(tmp_path / rel)
    (tmp_path / "List_of_testing_videos.txt").write_text(
        "1 Celeb-real/id0_0000.mp4\n1 YouTube-real/00001.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n")
    df = build_celebdf_manifest(str(tmp_path), True)
    assert len(df) == 3 and sorted(df.label) == [0, 0, 1]
    assert len(build_celebdf_manifest(str(tmp_path), False)) == 4
