"""Tests for git repository management utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infrastructure.github.git import _validate_git_url, _validate_org_name


class TestValidateOrgName:
    def test_valid_names(self) -> None:
        _validate_org_name("org", "name")
        _validate_org_name("my-org", "my-repo")
        _validate_org_name("org123", "name_123")
        _validate_org_name("org.name", "name.dot")

    def test_invalid_names(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            _validate_org_name("../org", "name")
        with pytest.raises(ValueError, match="path traversal"):
            _validate_org_name("org", "../name")
        with pytest.raises(ValueError, match="path traversal"):
            _validate_org_name("org/foo", "name")


class TestValidateGitUrl:
    def test_valid_https(self) -> None:
        _validate_git_url("https://github.com/org/repo.git")

    def test_valid_git(self) -> None:
        _validate_git_url("git://github.com/org/repo.git")

    def test_invalid_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_git_url("ftp://github.com/org/repo.git")

    def test_localhost_blocked(self) -> None:
        with pytest.raises(ValueError, match="localhost"):
            _validate_git_url("https://localhost/repo.git")

    def test_local_ip_blocked(self) -> None:
        with pytest.raises(ValueError, match="localhost"):
            _validate_git_url("https://127.0.0.1/repo.git")

    def test_private_ip_blocked(self) -> None:
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_git_url("https://10.0.0.1/repo.git")

    def test_private_ip_192_blocked(self) -> None:
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_git_url("https://192.168.1.1/repo.git")

    def test_hostname_allowed(self) -> None:
        _validate_git_url("https://github.com/org/repo.git")


class TestListReposOrgFilter:
    def test_list_repos_with_org_filter(self) -> None:
        """Test that org_filter limits repos to a specific org directory."""
        mock_repos_dir = Path("/tmp/mock_repos")
        mock_test_org = mock_repos_dir / "test-org"
        mock_other_org = mock_repos_dir / "other-org"

        mock_test_repo = mock_test_org / "repo-a"
        mock_other_repo = mock_other_org / "repo-b"

        mock_repo_a = {"active_branch": None}
        mock_repo_b = {"active_branch": None}

        with (
            patch(
                "os.path.isdir",
                side_effect=lambda x: (
                    x in (str(mock_repos_dir), str(mock_test_org), str(mock_other_org))
                ),
            ),
            patch("os.path.isfile", return_value=False),
            patch.object(
                Path,
                "iterdir",
                side_effect=[
                    [mock_test_org, mock_other_org],
                    [mock_test_repo, mock_other_repo],
                    [mock_test_repo],
                ],
            ),
            patch(
                "src.infrastructure.github.git.Repo",
                side_effect=[
                    mock_repo_a,
                    mock_repo_b,
                    mock_repo_a,
                    mock_repo_b,
                ],
            ),
            patch(
                "src.infrastructure.github.git.get_config",
                return_value=type("Config", (), {"paths_github_dir": str(mock_repos_dir)})(),
            ),
        ):
            from src.infrastructure.github.git import list_repos

            result = list_repos(repos_dir=mock_repos_dir, org_filter="test-org")

        # Should only include repos from test-org
        for repo in result:
            assert repo["org"] == "test-org"
