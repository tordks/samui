"""Azure Blob Storage service for image storage."""

import uuid
from io import BytesIO

from azure.storage.blob import BlobServiceClient, ContentSettings
from PIL import Image

from samui_backend.config import settings
from samui_backend.utils import get_image_content_type


class StorageService:
    """Service for managing image storage in Azure Blob Storage."""

    def __init__(self) -> None:
        """Initialize the storage service."""
        self._client = BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
        self._container_name = settings.azure_container_name
        self._ensure_container_exists()

    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist."""
        container_client = self._client.get_container_client(self._container_name)
        if not container_client.exists():
            container_client.create_container()

    def upload_image(self, file_content: bytes, filename: str) -> tuple[str, int, int]:
        """Upload an image to blob storage.

        Args:
            file_content: Raw image bytes.
            filename: Original filename.

        Returns:
            Tuple of (blob_path, width, height).
        """
        # Get image dimensions
        img = Image.open(BytesIO(file_content))
        width, height = img.size

        # Generate unique blob path
        extension = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        blob_path = f"images/{uuid.uuid4()}.{extension}"

        # Determine content type
        content_type = get_image_content_type(extension)

        # Upload to blob storage
        blob_client = self._client.get_blob_client(container=self._container_name, blob=blob_path)
        blob_client.upload_blob(
            file_content,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

        return blob_path, width, height

    def get_image(self, blob_path: str) -> bytes:
        """Download an image from blob storage.

        Args:
            blob_path: Path to the blob.

        Returns:
            Raw image bytes.
        """
        blob_client = self._client.get_blob_client(container=self._container_name, blob=blob_path)
        return blob_client.download_blob().readall()

    def delete_image(self, blob_path: str) -> None:
        """Delete an image from blob storage.

        Args:
            blob_path: Path to the blob.
        """
        blob_client = self._client.get_blob_client(container=self._container_name, blob=blob_path)
        blob_client.delete_blob(delete_snapshots="include")

    def upload_blob(self, blob_path: str, data: bytes, content_type: str | None = None) -> None:
        """Upload arbitrary data to blob storage.

        Args:
            blob_path: Path for the blob.
            data: Raw bytes to upload.
            content_type: Optional MIME content type.
        """
        blob_client = self._client.get_blob_client(container=self._container_name, blob=blob_path)
        settings = ContentSettings(content_type=content_type) if content_type else None
        blob_client.upload_blob(data, overwrite=True, content_settings=settings)
