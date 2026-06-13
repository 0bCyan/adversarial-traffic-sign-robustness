from streamlit.testing.v1 import AppTest


def test_demo_shows_checkpoint_message_when_model_is_missing() -> None:
    app = AppTest.from_file("src/demo/streamlit_app.py")
    app.run(timeout=10)
    assert not app.exception
    assert any("Missing model checkpoint" in item.value for item in app.error)
