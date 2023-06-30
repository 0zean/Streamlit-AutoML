import pandas as pd
import sqlalchemy
import streamlit as st
from db_con import create_connection


class DataFrameLoad:
    def __init__(self, data_source):
        self.data_source = data_source
        self.state = st.session_state

    def create_form(self):
       
        if self.data_source == "CSV File":
            self.state.clear()
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")

            if uploaded_file is not None:
                # Process the uploaded CSV file
                df = pd.read_csv(uploaded_file)
                if "df1" not in self.state:
                    self.state.df = df
                st.dataframe(df.head())
                st.success("CSV file uploaded successfully!", icon="🎉")

        elif self.data_source == "SQL Database":
            st.write("Please enter your SQL database details:")
            database_type = st.selectbox("Database Type", ("PostgreSQL", "MySQL"))
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            host = st.text_input("Host")
            port = st.text_input("Port")
            database_name = st.text_input("Database Name")

            if st.button("Connect"):
                # Clear Sesstion State on connect
                self.state.clear()
                try:
                    engine = create_connection(
                        username=username,
                        password=password,
                        host=host,
                        port=port,
                        db_name=database_name,
                        db_type=database_type.lower()
                    )

                    cursor = engine.connect()
                    st.success(f"Connected to {database_type} database!", icon="🎉")

                    # Store the database connection and cursor in the session self.state
                    if "engine" and "cursor" not in self.state:
                        self.state.engine = engine
                        self.state.cursor = cursor

                except sqlalchemy.exc.SQLAlchemyError:
                    st.error("Did not connect to Database!", icon="🚨")
                except ValueError:
                    st.warning("Make sure your information is correct!", icon="🚨")

            # Check if cursor is in Session self.state
            if hasattr(self.state, "cursor"):
                tables = sqlalchemy.inspect(self.state.engine).get_table_names()
                selected_table = st.selectbox("Select a table", tables)

                if st.button("Display Table"):
                    df = pd.read_sql_table(table_name=selected_table,
                                           con=self.state.cursor)
                    
                    if "df" not in self.state:
                        self.state.df = df
                    
                    st.dataframe(df.head())

        if "df" in self.state:
            features = st.multiselect("Which columns are your features?",
                                        options=list(self.state.df.columns))
            final_df = self.state.df[features]
            st.dataframe(final_df.head())

            self.state.final_df = final_df
    
    def retrieve_df(self):
        return self.state.final_df