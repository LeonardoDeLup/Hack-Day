import json

from fruit_backend.cli import main


def test_missing_file_is_json_on_stderr(tmp_path, capsys):
    assert main([str(tmp_path / "missing.jpg")]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"] == "file_error"


def test_invalid_image_is_json_on_stderr(tmp_path, capsys):
    image = tmp_path / "pretend.jpg"
    image.write_text("this is not a photo")
    assert main([str(image)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"] == "invalid_image"


def test_success_is_json_on_stdout(tmp_path, capsys, monkeypatch):
    class Result:
        def model_dump_json(self, indent):
            return '{"assessment": {"fruit_name": "banana"}}'

    image = tmp_path / "example.jpg"
    image.write_bytes(b"stub")
    monkeypatch.setattr("fruit_backend.cli.analyze_image", lambda data: Result())
    assert main([str(image)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["assessment"]["fruit_name"] == "banana"
