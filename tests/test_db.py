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


class TruncateTableTestCase(unittest.TestCase):
    """Keeping this test primitive - it's easy to go overboard."""
    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect_loop()

    def test_table_does_not_exist(self):
        """We should get a None return."""
        out = db.truncate_table(db_conn=self.conn, table='no_table_here')
        self.assertIsNone(out)

    def test_call_execute(self):
        with patch('MySQLdb.cursors.Cursor.execute', autospec=True) as p:
            db.truncate_table(db_conn=self.conn, table='bleh')

        p.assert_called_once()
        # As a class method, execute requires self as an input. So we'll
        # grab the second input and test it.
        self.assertEqual('TRUNCATE TABLE bleh', p.call_args[0][1])

    # def test_commit_called(self):
    #     with patch('MySQLdb.cursors.Cursor.execute', autospec=True):
    #         with patch.object(self.conn, attribute='commit', autospec=True) as p:
    #             db.truncate_table(db_conn=self.conn, table='some_table')
    #
    #     p.assert_called_once()

    def test_cursor_close_called(self):
        with patch('MySQLdb.cursors.Cursor.close', autospec=True) as p:
            db.truncate_table(db_conn=self.conn, table='other_table')

        p.assert_called_once()


class ExecuteAndFetchAllTestCase(unittest.TestCase):
    """Test execute_and_fetch_all"""
    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect_loop()

    def test_bad_query(self):
        with self.assertRaises(Error):
            db.execute_and_fetch_all(db_conn=self.conn, query='bad query')

    def test_fetch_all_called(self):
        with patch('MySQLdb.cursors.Cursor.fetchall', autospec=True) as p:
            db.execute_and_fetch_all(db_conn=self.conn,
                                     query='SHOW TABLES')

        # MySQLdb calls fetchall internally, so we'll expect two calls.
        self.assertEqual(p.call_count, 2)

    def test_cursor_closed(self):
        with patch('MySQLdb.cursors.Cursor.close', autospec=True) as p:
            db.execute_and_fetch_all(db_conn=self.conn,
                                     query='SHOW TABLES')

        p.assert_called_once()

    def test_get_output(self):
        """Assuming our database is setup."""
        out = db.execute_and_fetch_all(db_conn=self.conn,
                                       query='SHOW DATABASES')
        self.assertIn((os.environ['DB_DB'],), out)


if __name__ == '__main__':
    unittest.main()
