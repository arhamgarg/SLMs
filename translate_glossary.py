import sys

import torch
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"


def glossary_entries(pdf_path: str) -> list[tuple[str, str]]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    lines = [line.strip() for line in text.splitlines()]
    entries = [
        (term, definition)
        for term, definition in zip(lines, lines[1:])
        if term.replace("-", "").isalpha() and definition.endswith(".")
    ]
    if not entries:
        raise ValueError("No glossary entries were found in the PDF.")
    return entries


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "nlp_glossary_english.pdf"
    entries = glossary_entries(pdf_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    source = [f"{term}: {definition}" for term, definition in entries]
    encoded = {
        key: value.to(device)
        for key, value in tokenizer(
            source, return_tensors="pt", padding=True, truncation=True
        ).items()
    }
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=36, num_beams=1)
    translated = tokenizer.batch_decode(output, skip_special_tokens=True)

    print("NLP GLOSSARY: ENGLISH -> HINDI")
    print(f"Model: {MODEL_NAME}\n")
    for english, hindi in zip(source, translated, strict=True):
        print(f"EN  {english}")
        print(f"HI  {hindi}\n")


if __name__ == "__main__":
    main()
