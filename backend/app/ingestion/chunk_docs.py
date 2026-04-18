import os
import re
import tiktoken

# tokenizer for token count
tokenizer = tiktoken.get_encoding("cl100k_base")


def remove_frontmatter(text):
    """
    Removes YAML frontmatter from markdown
    """
    return re.sub(r"^---.*?---\n", "", text, flags=re.DOTALL)


def split_by_headings(text):
    """
    Split markdown by headings (#, ##, ###)
    """
    sections = re.split(r"\n#{1,6} ", text)
    return [s.strip() for s in sections if len(s.strip()) > 50]


def chunk_text(text, max_tokens=500):
    """
    Token-based chunking
    """
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append((chunk_text, len(chunk_tokens)))

    return chunks


def process_markdown_file(file_path):
    """
    Process a single markdown file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ❗ remove frontmatter
    text = remove_frontmatter(text)

    if len(text.strip()) < 50:
        return []

    sections = split_by_headings(text)

    chunks = []
    for section in sections:
        section_chunks = chunk_text(section)

        for chunk_text_, token_count in section_chunks:
            chunks.append({
                "text": chunk_text_,
                "token_count": token_count
            })

    return chunks


def chunk_documents(docs_path):
    """
    Main function to chunk all markdown files
    """
    all_chunks = []
    chunk_id = 0

    for root, _, files in os.walk(docs_path):
        print(f"Scanning: {root}")

        for file in files:
            if not file.endswith(".md"):
                continue

            file_path = os.path.join(root, file)
            print(f"Processing: {file}")

            file_chunks = process_markdown_file(file_path)

            for chunk in file_chunks:
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "source": file,
                    "section": root,
                    "token_count": chunk["token_count"]
                })
                chunk_id += 1

    return all_chunks