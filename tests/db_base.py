import pytest

from app.db import get_qdrant_client
from tests.base import TestBase


class TestDBBase(TestBase):
    @pytest.fixture(scope="function")
    def db_client(self, app):
        with app.app_context():
            client = get_qdrant_client()
        return client
