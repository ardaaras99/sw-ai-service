import sw_ai_service


def test_import() -> None:
    """Test that the package can be imported without errors."""
    assert isinstance(sw_ai_service.__name__, str)
