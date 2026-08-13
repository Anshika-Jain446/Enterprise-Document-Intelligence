import json
import os

import streamlit as st


class Register:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.user_file = (
            "users.json"
        )

    # ========================================================
    # LOAD USERS
    # ========================================================

    def load_users(self):

        if not os.path.exists(
            self.user_file
        ):

            return []

        try:

            with open(
                self.user_file,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        except Exception:

            return []

    # ========================================================
    # SAVE USERS
    # ========================================================

    def save_users(
        self,
        users
    ):

        with open(
            self.user_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                users,
                f,
                indent=4
            )

    # ========================================================
    # REGISTER USER
    # ========================================================

    def register_user(
        self,
        full_name,
        username,
        email,
        password,
        confirm_password
    ):

        users = (
            self.load_users()
        )

        if password != confirm_password:

            return (
                False,
                "Passwords do not match."
            )

        for user in users:

            if (
                user.get(
                    "username"
                )
                == username
            ):

                return (
                    False,
                    "Username already exists."
                )

            if (
                user.get(
                    "email"
                )
                == email
            ):

                return (
                    False,
                    "Email already exists."
                )

        users.append(
            {
                "full_name":
                    full_name,

                "username":
                    username,

                "email":
                    email,

                "password":
                    password,

                "role":
                    "User",
            }
        )

        self.save_users(
            users
        )

        return (
            True,
            "Account created successfully."
        )

    # ========================================================
    # REGISTER PAGE
    # ========================================================

    def register_page(self):

        st.title(
            "📝 Create Account"
        )

        full_name = st.text_input(
            "Full Name"
        )

        username = st.text_input(
            "Username"
        )

        email = st.text_input(
            "Email"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password"
        )

        if st.button(
            "Create Account"
        ):

            success, message = (
                self.register_user(
                    full_name,
                    username,
                    email,
                    password,
                    confirm_password
                )
            )

            if success:

                st.success(
                    message
                )

                st.info(
                    "You can now login."
                )

            else:

                st.error(
                    message
                )