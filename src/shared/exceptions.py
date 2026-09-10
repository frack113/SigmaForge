"""Custom exceptions for SigmaForge."""

from typing import Any


class SigmaError(Exception):
    """Base exception for Sigma errors."""

    http_status: int = 500

    def __init__(self, code: str, message: str, details: dict[str, str] | None = None) -> None:
        """Initialize exception."""
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ServiceUnavailableError(SigmaError):
    """Service unavailable error."""

    http_status = 503

    def __init__(self, service: str) -> None:
        """Initialize exception."""
        super().__init__(
            code="SERVICE_UNAVAILABLE",
            message=f"Service unavailable: {service}",
            details={"service": service},
        )


class ModelNotFoundError(SigmaError):
    """Model not found error."""

    http_status = 404

    def __init__(self, model_name: str) -> None:
        """Initialize exception."""
        super().__init__(
            code="MODEL_NOT_FOUND",
            message=f"Model not found: {model_name}",
            details={"model_name": model_name},
        )


class ValidationError(SigmaError):
    """Validation error."""

    http_status = 422

    def __init__(self, field: str, message: str) -> None:
        """Initialize exception."""
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            details={"field": field},
        )


class DownloadError(SigmaError):
    """Download error."""

    http_status = 500

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(
            code="DOWNLOAD_ERROR",
            message=message,
            details={},
        )


class UpdateError(SigmaError):
    """Update error."""

    http_status = 500

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(
            code="UPDATE_ERROR",
            message=message,
            details={},
        )


class BackupError(SigmaError):
    """Backup error."""

    http_status = 500

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(
            code="BACKUP_ERROR",
            message=message,
            details={},
        )


class ServiceStartError(Exception):
    """Raised when an auto-start service fails to start."""
