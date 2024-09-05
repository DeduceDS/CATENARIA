import pytest
from fastapi import UploadFile
from unittest.mock import Mock
from app.application.services import FileCheckerServiceImpl
from app.domain.exceptions.file_exceptions import InvalidFileFormatException


@pytest.mark.asyncio
async def test_validate_file_type_valid():
    mock_file = Mock()
    file = UploadFile(filename="test.pdf", file=mock_file)
    await FileCheckerServiceImpl.validate_file_type(file, ".pdf")
    # If no exception is raised, the test passes


@pytest.mark.asyncio
async def test_validate_file_type_invalid():
    mock_file = Mock()
    file = UploadFile(filename="test.txt", file=mock_file)
    with pytest.raises(InvalidFileFormatException):
        await FileCheckerServiceImpl.validate_file_type(file, ".pdf")


@pytest.mark.asyncio
async def test_validate_file_type_case_insensitive():
    mock_file = Mock()
    file = UploadFile(filename="test.PDF", file=mock_file)
    await FileCheckerServiceImpl.validate_file_type(file, ".pdf")
    # If no exception is raised, the test passes


@pytest.mark.asyncio
async def test_validate_file_type_no_extension():
    mock_file = Mock()
    file = UploadFile(filename="testfile", file=mock_file)
    with pytest.raises(InvalidFileFormatException):
        await FileCheckerServiceImpl.validate_file_type(file, ".pdf")
