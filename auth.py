import streamlit as st

from database import Database


class Authentication:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.db = Database()

    # ========================================================
    # LOGIN
    # ========================================================

    def login(
        self,
        username,
        password,
    ):

        username = username.strip()

        if not username:

            return (
                False,
                "Please enter your username."
            )

        if not password:

            return (
                False,
                "Please enter your password."
            )

        try:

            user = self.db.authenticate_user(
                username=username,
                password=password,
            )

            if not user:

                return (
                    False,
                    "Invalid Username or Password"
                )

            # --------------------------------------------
            # SESSION STATE
            # --------------------------------------------

            st.session_state.logged_in = True

            st.session_state.username = (
                user["username"]
            )

            st.session_state.role = (
                user["role"]
            )

            st.session_state.full_name = (
                user.get(
                    "full_name",
                    ""
                )
            )

            st.session_state.email = (
                user.get(
                    "email",
                    ""
                )
            )

            return (
                True,
                "Login successful."
            )

        except Exception as e:

            return (
                False,
                f"Login failed: {e}"
            )

    # ========================================================
    # LOGIN PAGE
    # ========================================================

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

        if st.button(
            "Login",
            use_container_width=True,
        ):

            success, message = self.login(
                username=username,
                password=password,
            )

            if success:

                st.success(
                    "✅ Login successful."
                )

                st.rerun()

            else:

                st.error(
                    message
                )

    # ========================================================
    # LOGOUT
    # ========================================================

    def logout(self):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.role = ""

        st.session_state.full_name = ""

        st.session_state.email = ""