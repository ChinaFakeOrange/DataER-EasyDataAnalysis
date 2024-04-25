# MIT License

# Copyright (c) 2024 ChinaFakeOrange

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# -*- ecoding: utf-8 -*-
# @Author: NUO


import streamlit as st
import requests
from streamlit_lottie import st_lottie


st.set_page_config(page_title='DataER', page_icon=None, layout="wide")
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def app():
    local_css("style/style.css")
    # ---- LOAD ASSETS ----
    lottie_coding = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_zzytykf2.json")

    # ---- HEADER SECTION ----
    with st.container():
        st.subheader("Hi,I am Your Data Analysis Assistant :wave:")
        st.write("**A Nobody**")
        st.write(
            "**Be Free to Data Usage.**"
        )
        st.write(
            "**Start Your Data Analysis From Left SideBar**"
        )

    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)

        contact_form = """
               <form action="https://formsubmit.co/xxxx@yourmail.com>" method="POST">
                   <input type="hidden" name="_captcha" value="false">
                   <input type="text" name="name" placeholder="Name" required>
                   <input type="email" name="email" placeholder="Email" required>
                   <textarea name="message" placeholder="Messages" required></textarea>
                   <button type="submit">Send</button>
               </form>
               """
        with left_column:
            st.header("Get In Touch~")
            st.write("##")
            st.markdown(contact_form, unsafe_allow_html=True)

        with right_column:
            st_lottie(lottie_coding, height=300, key="coding")

app()
