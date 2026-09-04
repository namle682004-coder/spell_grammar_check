import pytest
from unittest.mock import MagicMock

from src.storage.repositories.base_repo import BaseRepository
from src.storage.models.user import User


def test_base_repository_create():
    session = MagicMock()
    repo = BaseRepository(User, session)

    user = repo.create(email="a@b.com", username="alice", password_hash="hash")
    session.add.assert_called_once()
    session.flush.assert_called_once()
    assert user is not None
