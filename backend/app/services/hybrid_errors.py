"""Dependency-light exceptions shared by the hybrid inference services."""


class FoundationInferenceError(RuntimeError):
    pass


class FoundationNotReadyError(FoundationInferenceError):
    pass
