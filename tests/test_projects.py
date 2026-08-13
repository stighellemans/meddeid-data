import json

import pytest
from meddeid_language_nl.postprocess import post_process_spans

from meddeid_data.projects import (
    import_documents,
    init_project,
    package_annotation_set,
    split_project,
)


def test_txt_project_import_uses_private_stable_mapping_and_empty_spans(tmp_path):
    project = tmp_path / "project"
    source = tmp_path / "notes"
    source.mkdir()
    (source / "patient-123.txt").write_text("😀 Jan Peeters", encoding="utf-8")
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    artifact, manifest = import_documents(project, source)
    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["document_id"].startswith("doc-")
    assert "patient-123" not in artifact.read_text(encoding="utf-8")
    assert row["spans"] == [] and row["annotated"] is False
    assert manifest["contracts"]["offset_unit"] == "unicode_codepoints"
    assert (project / "private" / "source-map.jsonl").is_file()


def test_csv_import_rejects_duplicate_content(tmp_path):
    project = tmp_path / "project"
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    source = tmp_path / "notes.csv"
    source.write_text("id,text\na,Same note\nb,Same note\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate document content"):
        import_documents(project, source, id_column="id")


def test_csv_import_preserves_extra_and_structured_metadata(tmp_path):
    project = tmp_path / "project"
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    source = tmp_path / "notes.csv"
    source.write_text(
        'id,text,department,metadata_json\n'
        'a,"Jan kwam op controle",cardiologie,'
        '"{""patient"":{""given_name"":""Jan""},""known_values"":'
        '[{""value"":""Jan"",""label"":""Name:Patient""}]}"\n',
        encoding="utf-8",
    )
    artifact, manifest = import_documents(project, source, id_column="id")
    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["metadata"]["department"] == "cardiologie"
    assert row["metadata"]["patient"] == {"given_name": "Jan"}
    assert row["metadata"]["known_values"][0]["label"] == "Name:Patient"
    assert row["metadata"]["lang"] == "nl-BE"
    assert manifest["source"]["metadata"] == "auto"


def test_csv_import_can_select_or_discard_metadata(tmp_path):
    source = tmp_path / "notes.csv"
    source.write_text(
        "id,text,keep,drop\na,Unique note,yes,no\n", encoding="utf-8"
    )

    selected_project = tmp_path / "selected"
    init_project(selected_project, namespace="selected", language_profile="nl-BE")
    selected, _ = import_documents(
        selected_project, source, id_column="id", metadata_columns=["keep"]
    )
    selected_row = json.loads(selected.read_text(encoding="utf-8"))
    assert selected_row["metadata"]["keep"] == "yes"
    assert "drop" not in selected_row["metadata"]

    empty_project = tmp_path / "empty"
    init_project(empty_project, namespace="empty", language_profile="nl-BE")
    empty, _ = import_documents(
        empty_project, source, id_column="id", include_metadata=False
    )
    empty_row = json.loads(empty.read_text(encoding="utf-8"))
    assert set(empty_row["metadata"]) == {"lang", "source_content_sha256"}


def test_csv_import_maps_arbitrary_patient_and_repeated_caregiver_columns(tmp_path):
    project = tmp_path / "project"
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    source = tmp_path / "notes.csv"
    source.write_text(
        "id,text,voornaam,familienaam,auteur_1,auteur_2,department\n"
        'a,"Jan werd gezien door Anke De Vos en Bob Peeters",Jan,Janssens,'
        '"Anke De Vos",Bob Peeters,cardiologie\n',
        encoding="utf-8",
    )
    artifact, manifest = import_documents(
        project,
        source,
        id_column="id",
        patient_given_name_column="voornaam",
        patient_family_name_column="familienaam",
        caregiver_columns=["auteur_1", "auteur_2"],
    )
    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["metadata"]["patient"] == {
        "given_name": "Jan",
        "family_name": "Janssens",
    }
    assert row["metadata"]["caregivers"] == [
        {"given_name": "Anke De", "family_name": "Vos"},
        {"given_name": "Bob", "family_name": "Peeters"},
    ]
    assert row["metadata"]["department"] == "cardiologie"
    recovered = post_process_spans([], row["text"], metadata=row["metadata"])
    assert {(span["text"], span["label"]) for span in recovered} >= {
        ("Jan", "Name:Patient"),
        ("Anke De Vos", "Name:Caregiver"),
        ("Bob Peeters", "Name:Caregiver"),
    }
    assert manifest["source"]["name_mapping"]["caregiver_columns"] == [
        "auteur_1",
        "auteur_2",
    ]


def test_csv_import_splits_one_caregiver_column_on_literal_delimiter(tmp_path):
    project = tmp_path / "project"
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    source = tmp_path / "notes.csv"
    source.write_text(
        'id,text,patient,authors\na,"Mila zag Alice en Bob",Mila Janssens,'
        '"Alice Vermeulen; Bob Peeters; "\n',
        encoding="utf-8",
    )
    artifact, _ = import_documents(
        project,
        source,
        id_column="id",
        patient_name_column="patient",
        caregiver_columns=["authors"],
        caregiver_delimiter=";",
    )
    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["metadata"]["patient"] == {
        "given_name": "Mila",
        "family_name": "Janssens",
    }
    assert row["metadata"]["caregivers"] == [
        {"given_name": "Alice", "family_name": "Vermeulen"},
        {"given_name": "Bob", "family_name": "Peeters"},
    ]


def test_parquet_import_preserves_nested_metadata(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    project = tmp_path / "project"
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    source = tmp_path / "notes.parquet"
    table = pyarrow.Table.from_pylist(
        [
            {
                "note_id": 7,
                "note_text": "Mila kwam op controle",
                "department": "neurologie",
                "caregivers": ["Alice Vermeulen", "Bob Peeters"],
                "metadata": {"patient": {"given_name": "Mila"}},
            }
        ]
    )
    parquet.write_table(table, source)
    artifact, manifest = import_documents(
        project,
        source,
        id_column="note_id",
        text_column="note_text",
    )
    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["metadata"]["patient"]["given_name"] == "Mila"
    assert row["metadata"]["caregivers"] == [
        {"given_name": "Alice", "family_name": "Vermeulen"},
        {"given_name": "Bob", "family_name": "Peeters"},
    ]
    assert row["metadata"]["department"] == "neurologie"
    assert manifest["source"]["kind"] == "parquet"


def test_split_is_byte_identical_for_same_seed(tmp_path):
    project = tmp_path / "project"
    source = tmp_path / "notes"
    source.mkdir()
    for index in range(10):
        (source / f"{index}.txt").write_text(f"note {index}", encoding="utf-8")
    init_project(project, namespace="hospital-a", language_profile="en-GB")
    import_documents(project, source)
    first = split_project(project, seed=7, train_fraction=0.6, validation_fraction=0.2)
    first_bytes = (project / "manifests" / "splits.json").read_bytes()
    second = split_project(project, seed=7, train_fraction=0.6, validation_fraction=0.2)
    assert first == second
    assert first_bytes == (project / "manifests" / "splits.json").read_bytes()
    assert {name: value["documents"] for name, value in first["files"].items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_completed_annotation_set_gets_portable_identity_manifest(tmp_path):
    project = tmp_path / "project"
    init_project(project, namespace="hospital-a", language_profile="nl-BE")
    annotations = tmp_path / "reviewer-a.jsonl"
    annotations.write_text(
        json.dumps({
            "document_id": "doc-a",
            "text": "Jan",
            "spans": [{
                "begin": 0,
                "end": 3,
                "text": "Jan",
                "label": "Name:Patient",
                "category": "Name",
                "subtype": "Patient",
            }],
            "annotated": True,
        }) + "\n",
        encoding="utf-8",
    )
    manifest_path, manifest = package_annotation_set(
        project,
        annotations,
        annotation_set_id="hospital-a-round-1",
        annotator_id="reviewer-7",
    )
    assert manifest_path.name == "reviewer-a.manifest.json"
    assert manifest["files"]["annotations"] == "reviewer-a.jsonl"
    assert manifest["annotation_set_id"] == "hospital-a-round-1"
    assert manifest["contracts"]["offset_unit"] == "unicode_codepoints"
