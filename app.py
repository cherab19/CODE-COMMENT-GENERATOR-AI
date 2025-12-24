import streamlit as st
import transformers
import torch

# Cache the model and tokenizer so they load only once
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./model"
    # Resolve tokenizer and model classes with fallbacks for older transformers
    TokenizerCls = None
    ModelCls = None

    if hasattr(transformers, "AutoTokenizer"):
        TokenizerCls = transformers.AutoTokenizer
    elif hasattr(transformers, "T5Tokenizer"):
        TokenizerCls = transformers.T5Tokenizer
    elif hasattr(transformers, "PreTrainedTokenizerFast"):
        TokenizerCls = transformers.PreTrainedTokenizerFast

    if hasattr(transformers, "AutoModelForSeq2SeqLM"):
        ModelCls = transformers.AutoModelForSeq2SeqLM
    elif hasattr(transformers, "T5ForConditionalGeneration"):
        ModelCls = transformers.T5ForConditionalGeneration
    elif hasattr(transformers, "AutoModel"):
        ModelCls = transformers.AutoModel

    if TokenizerCls is None or ModelCls is None:
        raise ImportError(
            "No suitable tokenizer/model classes found in transformers. Please upgrade the transformers package."
        )

    tokenizer = TokenizerCls.from_pretrained(model_path)
    model = ModelCls.from_pretrained(model_path).to(device)
    return tokenizer, model, device

# Show loading message while model initializes
st.write("⏳ Loading model... please wait.")
tokenizer, model, device = load_model()
st.write("✅ Model loaded successfully!")

def generate_comment(code_snippet):
    input_text = "explain this python function in one short sentence: " + code_snippet
    # use tokenizer(...) to support both fast and legacy tokenizers
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=32,
            num_beams=2,   # reduced beams for faster response
            early_stopping=True
        )

    # Prefer tokenizer.decode if available, otherwise use batch_decode
    try:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception:
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

# Streamlit UI
st.title("📝 Code Comment Generator (T5-small)")
st.write("Paste a Python function below and get a one-line comment.")

code_input = st.text_area("Enter Python function:", height=200)

if st.button("Generate Comment"):
    if code_input.strip():
        comment = generate_comment(code_input)
        st.success(comment)
    else:
        st.warning("Please paste a Python function first.")
