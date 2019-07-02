import unittest
from unittest.mock import patch
from pyvvo import db
import os
from MySQLdb.connections import Connection
from MySQLdb._exceptions import Error

class DBEnvDefinedTestCase(unittest.TestCase):
    """Test the db_env_defined method."""
    def test_works(self):
        """This should work out of the box."""
        self.assertTrue(db.db_env_defined())

    def test_fail_db_host(self):
        copy = {**os.environ}
        del copy['DB_HOST']
        with patch.dict('os.environ', copy, clear=True):
            self.assertFalse(db.db_env_defined())

    def test_fail_db_user(self):
        copy = {**os.environ}
        del copy['DB_USER']
        with patch.dict('os.environ', copy, clear=True):
            self.assertFalse(db.db_env_defined())

    def test_fail_db_pass(self):
        copy = {**os.environ}
        del copy['DB_PASS']
        with patch.dict('os.environ', copy, clear=True):
            self.assertFalse(db.db_env_defined())

    def test_fail_db_db(self):
        copy = {**os.environ}
        del copy['DB_DB']
        with patch.dict('os.environ', copy, clear=True):
            self.assertFalse(db.db_env_defined())

    def test_fail_db_port(self):
        copy = {**os.environ}
        del copy['DB_PORT']
        with patch.dict('os.environ', copy, clear=True):
            self.assertFalse(db.db_env_defined())


class ConnectLoopTestCase(unittest.TestCase):
    """Test connect_loop method."""

    def test_works(self):
        """This should work out of the box."""
        c = db.connect_loop()
        self.assertIsInstance(c, Connection)

    def test_bad_password(self):
        copy = os.environ.copy()
        copy['DB_PASS'] = ''
        with patch.dict('os.environ', copy, clear=True):
            with self.assertRaises(Error):
                db.connect_loop(timeout=0.01, retry_interval=0.01)

    def test_bad_db(self):
        copy = os.environ.copy()
        copy['DB_DB'] = 'some_database'
        with patch.dict('os.environ', copy, clear=True):
            with self.assertRaises(Error):
                db.connect_loop(timeout=0.01, retry_interval=0.01)

    def test_bad_host(self):
        copy = os.environ.copy()
        copy['DB_HOST'] = 'some_host'
        with patch.dict('os.environ', copy, clear=True):
            with self.assertRaises(Error):
                db.connect_loop(timeout=0.01, retry_interval=0.01)

    def test_bad_user(self):
        copy = os.environ.copy()
        copy['DB_USER'] = 'some_user'
        with patch.dict('os.environ', copy, clear=True):
            with self.assertRaises(Error):
                db.connect_loop(timeout=0.01, retry_interval=0.01)

    def test_bad_port(self):
        copy = os.environ.copy()
        copy['DB_PORT'] = '1234'
        with patch.dict('os.environ', copy, clear=True):
            with self.assertRaises(Error):
                db.connect_loop(timeout=0.01, retry_interval=0.01)


if __name__ == '__main__':
    unittest.main()
