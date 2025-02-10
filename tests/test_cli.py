import pytest

from ai_service.db import COLLECTION_NAME, get_qdrant_client
from tests.base import TestBase


class TestCLI(TestBase):
    @pytest.fixture(scope="class")
    def runner(self, app):
        return app.test_cli_runner()

    def test_populate_db(self, runner, app):  # Add app fixture
        runner.invoke(args="populate_db")
        with app.app_context():  # Wrap DB client usage in app context
            client = get_qdrant_client()
            count = client.count(
                collection_name=COLLECTION_NAME,
            ).count
            assert count == 100
            search_result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_vectors=True,
            )[0]
            for record in search_result:
                assert abs(1 - sum(record.vector)) < 1e-5
                assert len(record.payload) > 0
