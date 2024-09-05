# app/domain/exceptions/file_exceptions.py
class InvalidFileFormatException(Exception):
    def __init__(self, message: str = "Invalid file format"):
        self.message = message
        super().__init__(self.message)


class FileProcessingException(Exception):
    def __init__(self, message: str = "Error processing the file"):
        self.message = message
        super().__init__(self.message)
