import app

class TestApp:
    def test_main_completes(self):
        """
        This test method checks that the main method runs to completion, returning exit code 0.
        """
        assert app.main() == 0