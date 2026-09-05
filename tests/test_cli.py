import json

import pytest

from meddeid_data.cli import main


def test_project_create_imports_and_prints_preannotation_handoff(tmp_path, capsys):
    source = tmp_path / "notes.csv"
    source.write_text(
        'note_id,note_text,metadata_json\n'
        'n-1,"Jan kwam op controle","{""patient"":'
        '{""given_name"":""Jan""}}"\n',
        encoding="utf-8",
    )
    project = tmp_path / "project"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
            "--id-column",
            "note_id",
            "--text-column",
            "note_text",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "annotation-ready documents" in output
    assert "meddeid batch" in output
    assert "MEDDEID_ANNOTATIONS_PATH" in output
    row = json.loads((project / "artifacts" / "annotations.jsonl").read_text())
    assert row["metadata"]["patient"]["given_name"] == "Jan"


def test_project_create_prints_paths_relative_to_current_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")

    assert main(
        [
            "project",
            "create",
            "validation",
            "notes.csv",
            "--namespace",
            "hospital-study",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Created validation with 1 annotation-ready documents" in output
    assert "Canonical dataset: validation/artifacts/annotations.jsonl" in output
    assert "Manifest: validation/manifests/input-documents.json" in output
    assert "meddeid batch validation/artifacts/annotations.jsonl" in output
    assert "--output validation/assignments/model-assisted-review.jsonl" in output
    assert "MEDDEID_ANNOTATIONS_PATH=/input/model-assisted-review.jsonl" in output
    assert (
        '-v "$PWD/validation/assignments/model-assisted-review.jsonl:'
        '/input/model-assisted-review.jsonl"'
        in output
    )
    assert str(tmp_path) not in output


def test_project_create_keeps_absolute_paths_outside_current_directory(
    tmp_path, monkeypatch, capsys
):
    working_directory = tmp_path / "elsewhere"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    source = tmp_path / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")
    project = tmp_path / "validation"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert f"Canonical dataset: {project}/artifacts/annotations.jsonl" in output
    assert f"meddeid batch {project}/artifacts/annotations.jsonl" in output


def test_english_project_does_not_recommend_dutch_preannotation(tmp_path, capsys):
    source = tmp_path / "notes.csv"
    source.write_text('note_id,text\nn-1,"English clinical note"\n', encoding="utf-8")
    project = tmp_path / "english-project"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "english-study",
            "--language-profile",
            "en-GB",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "No released local pre-annotation model is configured for en-GB" in output
    assert "meddeid-dutch-synth" not in output
    row = json.loads((project / "artifacts" / "annotations.jsonl").read_text())
    assert row["metadata"]["lang"] == "en-GB"


def test_annotation_handoff_detects_configured_source_checkout(
    tmp_path, monkeypatch, capsys
):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    checkout = tmp_path / "meddeid-annotate"
    checkout.mkdir()
    (checkout / "package.json").write_text(
        '{"name":"meddeid-annotate"}\n', encoding="utf-8"
    )
    monkeypatch.setenv("MEDDEID_ANNOTATE_DIR", str(checkout))
    source = working_directory / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")

    assert main(
        [
            "project",
            "create",
            "validation",
            "notes.csv",
            "--namespace",
            "hospital-study",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "A local meddeid-annotate checkout was detected" in output
    assert f"npm ci --prefix {checkout}" in output
    assert f"npm --prefix {checkout} run dev" in output
    assert "docker run" not in output


def test_annotation_handoff_explains_non_pip_docker_route(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEDDEID_ANNOTATE_DIR", raising=False)
    monkeypatch.setattr("meddeid_data.cli.shutil.which", lambda command: None)
    source = tmp_path / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")

    assert main(
        [
            "project",
            "create",
            "validation",
            "notes.csv",
            "--namespace",
            "hospital-study",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "requires the MedDeID inference package" in output
    assert "python -m pip install 'meddeid>=0.3,<0.4'" in output
    assert "meddeid-annotate is a separate application, not a pip package" in output
    assert "install Docker Desktop or Docker Engine first" in output
    assert "downloaded automatically on first use" in output
    assert "ghcr.io/stighellemans/meddeid-annotate:0.2.0" in output


def test_project_create_accepts_common_name_column_mappings(tmp_path):
    source = tmp_path / "notes.csv"
    source.write_text(
        'id,body,patient,authors\nn-1,"Jan zag Alice en Bob",Jan Peeters,'
        '"Alice Vermeulen|Bob Janssens"\n',
        encoding="utf-8",
    )
    project = tmp_path / "project"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
            "--id-column",
            "id",
            "--text-column",
            "body",
            "--patient-name-column",
            "patient",
            "--caregiver-column",
            "authors",
            "--caregiver-delimiter",
            "|",
        ]
    ) == 0
    row = json.loads((project / "artifacts" / "annotations.jsonl").read_text())
    assert len(row["metadata"]["caregivers"]) == 2


def test_project_create_autodetects_canonical_data_scientist_columns(tmp_path):
    source = tmp_path / "notes.csv"
    source.write_text(
        "source_id,text,patient_given_name,patient_family_name,"
        "caregiver_1_given_name,caregiver_1_family_name,"
        "caregiver_2_first_name,caregiver_2_last_name\n"
        'n-1,"Jan zag Alice en Bob",Jan,Peeters,Alice,Vermeulen,Bob,Janssens\n',
        encoding="utf-8",
    )
    project = tmp_path / "project"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
        ]
    ) == 0
    row = json.loads((project / "artifacts" / "annotations.jsonl").read_text())
    assert row["metadata"]["patient"] == {
        "given_name": "Jan",
        "family_name": "Peeters",
    }
    assert row["metadata"]["caregivers"] == [
        {"given_name": "Alice", "family_name": "Vermeulen"},
        {"given_name": "Bob", "family_name": "Janssens"},
    ]
    saved = json.loads((project / "manifests" / "import-mapping.json").read_text())
    assert saved["id_column"] == "source_id"
    assert saved["caregivers"] == [
        {
            "given_name_column": "caregiver_1_given_name",
            "family_name_column": "caregiver_1_family_name",
        },
        {
            "given_name_column": "caregiver_2_first_name",
            "family_name_column": "caregiver_2_last_name",
        },
    ]


def test_project_create_retry_continues_untouched_scaffold(tmp_path, capsys):
    source = tmp_path / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")
    project = tmp_path / "project"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
            "--text-column",
            "wrong_column",
        ]
    ) == 2
    key_before = (project / "private" / "document-id.key").read_text()
    error = capsys.readouterr().err
    assert "Choose the correct value for --text-column" in error
    assert "Available columns: 'source_id', 'text'" in error

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Continuing existing empty MedDeID project" in output
    assert "Completed" in output
    assert (project / "private" / "document-id.key").read_text() == key_before
    assert (project / "artifacts" / "annotations.jsonl").is_file()


def test_project_create_suggests_correct_text_column_in_recovery_command(
    tmp_path, capsys
):
    source = tmp_path / "notes.csv"
    source.write_text(
        "source_id,text,document_creation_date\n"
        "n-1,Unique note,2026-08-31\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"

    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
            "--text-column",
            "text_id",
            "--id-column",
            "source_id",
        ]
    ) == 2

    error = capsys.readouterr().err
    assert "--text-column value 'text_id' does not match any source column" in error
    assert "Did you mean 'text'?" in error
    assert "Available columns: 'source_id', 'text', 'document_creation_date'" in error
    recovery = error.split("If 'text' is the intended column, run:\n", 1)[1]
    assert "--text-column text" in recovery
    assert "--text-column text_id" not in recovery


def test_project_create_retry_refuses_project_with_data(tmp_path):
    source = tmp_path / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")
    project = tmp_path / "project"
    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "hospital-study",
        ]
    ) == 0

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "project",
                "create",
                str(project),
                str(source),
                "--namespace",
                "hospital-study",
            ]
        )


def test_project_create_retry_refuses_identity_change(tmp_path):
    source = tmp_path / "notes.csv"
    source.write_text("source_id,text\nn-1,Unique note\n", encoding="utf-8")
    project = tmp_path / "project"
    assert main(
        [
            "project",
            "create",
            str(project),
            str(source),
            "--namespace",
            "first-study",
            "--text-column",
            "wrong_column",
        ]
    ) == 2

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "project",
                "create",
                str(project),
                str(source),
                "--namespace",
                "different-study",
            ]
        )


def test_mapping_config_is_saved_and_reused_on_later_import(tmp_path):
    mapping = tmp_path / "hospital-export.yaml"
    mapping.write_text(
        "\n".join(
            [
                "version: meddeid.import-mapping.v1",
                "text_column: note_body",
                "id_column: export_key",
                "patient_name_column: subject",
                "caregivers:",
                "  - given_name_column: author_first",
                "    family_name_column: author_last",
                "  - given_name_column: coauthor_first",
                "    family_name_column: coauthor_last",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.csv"
    first.write_text(
        "export_key,note_body,subject,author_first,author_last,"
        "coauthor_first,coauthor_last\n"
        'a,"Mila zag Alice en Bob",Mila Peeters,Alice,Vermeulen,Bob,Janssens\n',
        encoding="utf-8",
    )
    project = tmp_path / "project"
    assert main(
        [
            "project",
            "create",
            str(project),
            str(first),
            "--namespace",
            "hospital-study",
            "--mapping-config",
            str(mapping),
        ]
    ) == 0

    second = tmp_path / "second.csv"
    second.write_text(
        "export_key,note_body,subject,author_first,author_last,"
        "coauthor_first,coauthor_last\n"
        'b,"Jan zag Chris en Dana",Jan Peeters,Chris,Janssens,Dana,Peeters\n',
        encoding="utf-8",
    )
    assert main(
        ["project", "import", str(project), str(second)]
    ) == 0
    row = json.loads((project / "artifacts" / "annotations.jsonl").read_text())
    assert row["metadata"]["patient"]["given_name"] == "Jan"
    assert row["metadata"]["caregivers"] == [
        {"given_name": "Chris", "family_name": "Janssens"},
        {"given_name": "Dana", "family_name": "Peeters"},
    ]
