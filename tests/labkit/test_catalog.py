"""Every lab under labs/ loads, and each task names a check its check.py defines."""

from pathlib import Path

import pytest

from labkit.catalog import LabError, load_catalog, load_checks, load_lab

LABS = Path(__file__).resolve().parents[2] / "labs"


def test_catalog_finds_labs():
    labs = load_catalog(LABS)
    assert labs, "no labs found"
    assert [lab.id for lab in labs] == sorted(lab.id for lab in labs)


@pytest.mark.parametrize("lab", load_catalog(LABS), ids=lambda lab: lab.id)
def test_lab_is_complete(lab):
    assert lab.id == lab.path.name
    assert lab.tasks and lab.interview_questions and lab.skills
    assert lab.fidelity.real, "say what is real"
    assert lab.fidelity.differs_on_aws, "say what differs on real AWS"
    assert (lab.path / "README.md").is_file()
    checks = load_checks(lab)
    for task in lab.tasks:
        assert task.check in checks, f"{task.id}: no check named {task.check!r}"


def test_bad_lab_names_the_file(tmp_path):
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "lab.toml").write_text('id = "broken"\n')
    with pytest.raises(LabError, match="lab.toml"):
        load_lab(bad)


def test_id_must_match_folder(tmp_path):
    lab = tmp_path / "folder"
    lab.mkdir()
    src = (LABS / "lambda-sqs-pipeline" / "lab.toml").read_text()
    (lab / "lab.toml").write_text(src)
    with pytest.raises(LabError, match="folder"):
        load_lab(lab)
