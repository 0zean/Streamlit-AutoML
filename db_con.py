from sqlalchemy import create_engine


def create_connection(username, password, host, port, 
                      db_name, db_type, driver='pymysql'):
    """Connects to a database using the given credentials for
    authentication and returns the session object

    Parameters
    ----------
    username : _str_
        Username for database
    password : _str_
        Password for database
    host : _str_
        Host IP or hostname for database
    port : _str_
        Port database is exposed on
    db_name : _str_
        Database name, can leave blank
    db_type : _str_
        Type of database, (PostgreSQL or MySQL)
    driver : _str, optional_
        Driver to use with MySQL, by default 'pymysql'

    Returns
    -------
    _obj_
        Session object (engine)

    Raises
    ------
    ValueError
        If an invalid database type is provided
    """
    # Create the Dialect for Postgres or MySQl
    if db_type == 'postgresql':
        dialect = f'postgresql://{username}:{password}@{host}:{port}/{db_name}'
    elif db_type == 'mysql':
        dialect = f'mysql+{driver}://{username}:{password}@{host}:{port}/{db_name}'
    else:
        raise ValueError(f'{db_type} not supported type: postgresql, mysql')

    # Use the create_engine method
    engine = create_engine(dialect)

    return engine
