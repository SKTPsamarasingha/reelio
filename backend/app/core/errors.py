class ApiError(Exception):
    """Raise this anywhere; it becomes {'error': {'code': ..., 'message': ...}}."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class UserFacingError(ApiError):
    """A convenience subclass for standard 400 Bad Request errors shown to the user."""

    def __init__(self, message: str, code: str = "BAD_REQUEST"):
        super().__init__(status_code=400, code=code, message=message)
