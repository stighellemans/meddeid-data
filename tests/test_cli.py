import json

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
