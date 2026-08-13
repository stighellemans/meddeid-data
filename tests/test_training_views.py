import json

import pytest

from meddeid_data.projects import import_documents, init_project, split_project
from meddeid_data.training_views import prepare_training_views


def _read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write(path, rows):
    path.write_text(
        "".join(json.dumps({**row, "annotated": True}) + "\n" for row in rows),
        encoding="utf-8",
    )


def _project(tmp_path):
    project = tmp_path / "project"
    source = tmp_path / "notes"
    source.mkdir()
    for index in range(5):
        (source / f"{index}.txt").write_text(f"unique note {index}", encoding="utf-8")
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    import_documents(project, source)
    split_project(project, seed=42, train_fraction=0.4, validation_fraction=0.2)
    return project


def test_prepare_training_views_recombines_development_and_withholds_test(tmp_path):
    project = _project(tmp_path)
    train = tmp_path / "reviewed-train.jsonl"
    validation = tmp_path / "reviewed-validation.jsonl"
    test = tmp_path / "test-gold.jsonl"
    _write(train, _read(project / "splits" / "train.jsonl"))
    _write(validation, _read(project / "splits" / "validation.jsonl"))
    _write(test, _read(project / "splits" / "test.jsonl"))

    output, summary = prepare_training_views(
        project,
        selection_train=train,
        selection_validation=validation,
        test_gold=test,
    )

    assert summary["development_documents"] == 3
    assert summary["test_documents"] == 2
    assert len(_read(output / "fit" / "train.jsonl")) == 2
    assert len(_read(output / "fit" / "val.jsonl")) == 1
    assert len(_read(output / "fit" / "test.jsonl")) == 2
    assert len(_read(output / "selection" / "train.jsonl")) == 2
    assert len(_read(output / "selection" / "val.jsonl")) == 1
    assert (output / "selection" / "test.jsonl").read_text() == ""
    assert len(_read(output / "refit" / "train.jsonl")) == 3
    assert len(_read(output / "refit" / "val.jsonl")) == 1
    assert len(_read(output / "refit" / "test.jsonl")) == 2
    selection_manifest = json.loads(
        (output / "selection" / "manifest.json").read_text()
    )
    assert selection_manifest["phase"] == "epoch_selection"
    assert selection_manifest["files"]["test"]["documents"] == 0
    assert len(selection_manifest["files"]["train"]["sha256"]) == 64


def test_prepare_training_views_accepts_one_combined_development_file(tmp_path):
    project = _project(tmp_path)
    development = tmp_path / "reviewed-development.jsonl"
    test = tmp_path / "test-gold.jsonl"
    _write(
        development,
        [
            *_read(project / "splits" / "validation.jsonl"),
            *_read(project / "splits" / "train.jsonl"),
        ],
    )
    _write(test, _read(project / "splits" / "test.jsonl"))

    output, summary = prepare_training_views(
        project,
        development=development,
        test_gold=test,
    )

    assert summary["development_documents"] == 3
    assert len(_read(output / "fit" / "train.jsonl")) == 2
    assert len(_read(output / "fit" / "val.jsonl")) == 1
    fit_manifest = json.loads((output / "fit" / "manifest.json").read_text())
    assert fit_manifest["phase"] == "single_fit"
    assert set(fit_manifest["sources"]) >= {"development", "test_gold"}


def test_prepare_training_views_rejects_incomplete_or_wrong_assignments(tmp_path):
    project = _project(tmp_path)
    train = tmp_path / "reviewed-train.jsonl"
    validation = tmp_path / "reviewed-validation.jsonl"
    test = tmp_path / "test-gold.jsonl"
    train.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in _read(project / "splits" / "train.jsonl")
        ),
        encoding="utf-8",
    )
    _write(validation, _read(project / "splits" / "validation.jsonl"))
    _write(test, _read(project / "splits" / "test.jsonl"))

    with pytest.raises(ValueError, match="not marked annotated/completed"):
        prepare_training_views(
            project,
            selection_train=train,
            selection_validation=validation,
            test_gold=test,
        )

    _write(train, _read(project / "splits" / "train.jsonl"))
    wrong_test = _read(test)
    wrong_test[0]["text"] += " changed"
    _write(test, wrong_test)
    with pytest.raises(ValueError, match="test gold text differs"):
        prepare_training_views(
            project,
            selection_train=train,
            selection_validation=validation,
            test_gold=test,
        )
