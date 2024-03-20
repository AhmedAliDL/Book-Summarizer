import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline

st.set_page_config(
    page_title="Book Summarizer",
    page_icon="🤖",
    layout="wide",
)

# remove made with streamlit and hamburger
st.markdown("""
<style>
.css-fblp2m.ex0cdmw0
{
  visibility:hidden;
}
.css-10pw50.egzxvld1
{
 visibility:hidden;
}
<style/>
""", unsafe_allow_html=True)


def read_files(pdf_docs):
    text: str = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_chunks(text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=10,
        length_function=len

    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_summarize_text(chunks):
    text = ""
    for chunk in chunks:
        text += f"{st.session_state.summarizer(chunk)[0]['summary_text']}\n"
    return text


def main():
    st.title("Book Summarizer")
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device="cuda")
    files = st.file_uploader("Upload book:", type=["pdf"], accept_multiple_files=True)
    gen = st.button("Generate")
    if files:
        book_summarized = {}
        if gen:
            with st.spinner():
                for file in files:
                    file_name = file.name.split('.')[0]
                    book_summarized[file_name] = get_summarize_text(
                        get_chunks(
                            read_files(
                                file)))

        if book_summarized:
            for book_name, summarized_text in book_summarized.items():
                st.subheader(book_name)
                st.text(summarized_text)


if __name__ == "__main__":
    main()
