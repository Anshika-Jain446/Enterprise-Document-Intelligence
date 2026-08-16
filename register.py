import streamlit as st

from database import Database


class Register:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.db = Database()

    # ========================================================
    # REGISTER USER
    # ========================================================

    def register_user(
        self,
        full_name,
        username,
        email,
        password,
        confirm_password,
    ):

        full_name = full_name.strip()
        username = username.strip()
        email = email.strip()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not full_name:

            return (
                False,
                "Please enter your full name."
            )

        if not username:

            return (
                False,
                "Please enter a username."
            )

        if not email:

            return (
                False,
                "Please enter your email."
            )

        if not password:

            return (
                False,
                "Please enter a password."
            )

        if password != confirm_password:

            return (
                False,
                "Passwords do not match."
            )

        if len(password) < 6:

            return (
                False,
                "Password must contain at least 6 characters."
            )

        # ----------------------------------------------------
        # CHECK EXISTING USER
        # ----------------------------------------------------

        try:

            if self.db.user_exists(
                username=username,
                email=email,
            ):

                return (
                    False,
                    "Username or email already exists."
                )

            # ------------------------------------------------
            # CREATE USER IN POSTGRESQL
            # ------------------------------------------------

            user = self.db.create_user(
                username=username,
                full_name=full_name,
                email=email,
                password=password,
                role="User",
            )

            if user:

                return (
                    True,
                    "Account created successfully."
                )

            return (
                False,
                "Account could not be created."
            )

        except Exception as e:

            return (
                False,
                f"Registration failed: {e}"
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
            "Create Account",
            use_container_width=True,
        ):

            success, message = self.register_user(
                full_name=full_name,
                username=username,
                email=email,
                password=password,
                confirm_password=confirm_password,
            )

            if success:

                st.success(
                    f"✅ {message}"
                )

                st.info(
                    "You can now go to Login."
                )

            else:

                st.error(
                    message
                )