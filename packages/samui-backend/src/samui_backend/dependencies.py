"""FastAPI dependency providers for shared services."""

from samui_backend.services import SAM3Service, StorageService

_storage_service: StorageService | None = None
_sam3_service: SAM3Service | None = None


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


def get_sam3_service() -> SAM3Service:
    """Get or create the SAM3 service singleton."""
    global _sam3_service
    if _sam3_service is None:
        _sam3_service = SAM3Service()
    return _sam3_service
