import json
import os

import streamlit as st


class Authentication:

    def __init__(self):

        self.user_file = "users.json"

    def load_users(self):

        if not os.path.exists(
            self.user_file
        ):

            return []

        with open(
            self.user_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def login(
        self,
        username,
        password
    ):

        users = self.load_users()

        for user in users:

            if (
                user["username"] == username
                and
                user["password"] == password
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                st.session_state.role = user["role"]

                return True

        return False

    def logout(self):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.role = ""

    def login_page(self):

        st.title(
            "🔐 Enterprise Login"
        )

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if self.login(
                username,
                password
            ):

                st.success(
                    "Login Successful"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid Username or Password"
                )