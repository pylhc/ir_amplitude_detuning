from __future__ import annotations

import logging
import sys
from io import StringIO

import pytest

from ir_amplitude_detuning.utilities.logging import log_setup


class TestLogSetup:
    """Test cases for the log_setup function."""

    def test_log_setup(self):
        """Test that log_setup sets logging level to INFO."""
        logger = logging.getLogger()
        logger.handlers = []  # remove pytest handlers

        log_setup()

        # Verify logging is configured
        assert len(logger.handlers) > 0

        # Check that at least one handler has INFO level or root logger is INFO
        assert logger.level == logging.INFO or any(
            h.level == logging.INFO for h in logger.handlers
        )

        # Verify a StreamHandler exists pointing to stdout
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        ]
        assert len(stream_handlers) > 0

    def test_log_setup_format_string(self):
        """Test that log_setup applies correct format string."""
        logger = logging.getLogger()
        logger.handlers = []  # remove pytest handlers

        log_setup()

        # Get the formatter from the handler
        handler = logger.handlers[0]
        formatter = handler.formatter

        assert formatter is not None
        assert "%(levelname)7s" in formatter._fmt
        assert "%(message)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt

    def test_log_setup_logging_works(
        self, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
    ):
        """Test that logging actually works after setup."""
        logger = logging.getLogger()
        caplog_handlers = logger.handlers[-2:]
        logger.handlers = []  # remove pytest handlers

        # Test before setup -> no output to stdout
        msg = "Test message"
        logger.info(msg)
        assert msg not in capsys.readouterr()[0]

        # run setup with force=true to override caplog setup
        log_setup(force=True)

        # add caplog handlers back, so it can do it's job
        for handler in caplog_handlers:
            logger.addHandler(handler)

        # Test that message is in caplog and std
        logger.info(msg)

        assert msg in capsys.readouterr()[0]
        assert msg in caplog.text
