"""Module for database related methods.

The following environment variables will be used for database
connections:
    DB_HOST: hostname of the MySQL database.
    DB_USER: User to use to connect to MySQL.
    DB_PASS: Password for DB_USER.
    DB_DB: Database to use.
    DB_PORT: Port to connect to MySQL.
"""
import MySQLdb
import time
import os


def db_env_defined():
    """Helper to check if the necessary database-related environment
    variables are all present.
    """
    try:
        _ = os.environ['DB_HOST']
        _ = os.environ['DB_USER']
        _ = os.environ['DB_PASS']
        _ = os.environ['DB_DB']
        _ = os.environ['DB_PORT']
    except KeyError:
        return False
    else:
        return True


def connect_loop(timeout=30, retry_interval=0.25):
    """Attempt to connect to the database for a set amount of time.

    :param timeout: Total time (seconds) to wait for the database.
    :param retry_interval: Time (seconds) between reconnection attempts.

    :return MySQLdb.connections.Connection object.

    :raises MySQLdb.Error

    Note that this method is primitive: It isn't tracking the entire
    function runtime, but rather sleeping for retry_interval in
    between attempts. Thus, the maximum number of attempts is exactly
    timeout / retry_interval.

    Note:
    """
    # Initialize time counter.
    t = 0
    while t < timeout:
        try:
            c = MySQLdb.connect(db=os.environ['DB_DB'],
                                passwd=os.environ['DB_PASS'],
                                host=os.environ['DB_HOST'],
                                user=os.environ['DB_USER'],
                                port=int(os.environ['DB_PORT']))
        except MySQLdb.Error as e:
            # Assign most recent error so we can raise it at the end.
            recent_e = e
            # Wait for retry_interval seconds.
            time.sleep(retry_interval)
            # Bump our time counter.
            t += retry_interval
        else:
            # We've successfully connected.
            return c

    # We're outside of the loop, raise exception.
    # noinspection PyUnboundLocalVariable
    raise recent_e


def truncate_table(db_conn, table):
    """Truncate a table if it exists."""
    # Get a cursor.
    cursor = db_conn.cursor()
    try:
        # Run the command.
        cursor.execute("TRUNCATE TABLE " + table)
    except db_conn.ProgrammingError as e:
        # It's okay if the table doesn't exist.
        expected_string = "Table '{}.{}' doesn't exist".format(
            os.environ['DB_DB'], table
        )
        if not ((e.args[0] == 1146) and (e.args[1] == expected_string)):
            # We got some other error, so we should raise it.
            raise e
    else:
        # If the command succeeded, call commit. This may not be
        # necessary.
        db_conn.commit()
    finally:
        # Always close the cursor.
        cursor.close()


def execute_and_fetch_all(db_conn, query):
    """Simple wrapper to execute a query and return all results."""
    # Get cursor.
    cursor = db_conn.cursor()
    try:
        cursor.execute(query)
    except Exception as e:
        raise e
    else:
        # Fetch the data.
        out = cursor.fetchall()
    finally:
        cursor.close()
        db_conn.commit()

    return out
