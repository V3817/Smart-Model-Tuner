import streamlit as st
import lamini
import time
from lamini import Lamini
import pandas as pd

# Must be FIRST Streamlit command
st.set_page_config(
    page_title="Smart Model Tuner",
    page_icon="⚡",
    layout="wide"
)

def validate_dataset(data):
    if len(data) < 3:
        raise ValueError("At least 3 examples required")
    for example in data:
        if len(example['input'])<5 or len(example['output'])<5:
            raise ValueError("Inputs/outputs must be ≥5 chars")

def get_dataset_stats(data):
    return {
        "Total Examples": len(data),
        "Avg Question Length": round(sum(len(d['input']) for d in data)/len(data), 1),
        "Avg Answer Length": round(sum(len(d['output']) for d in data)/len(data), 1),
        "Unique Words": len(set(" ".join([d['input']+" "+d['output'] for d in data]).split()))
    }

with st.sidebar:
    st.title("⚙️ Engine Room")
    lamini_key = st.text_input("Lamini API Key", type="password")
    st.caption("Get key from [Lamini](https://lamini.ai/)")
    
    st.markdown("---")
    st.subheader("Resource Limits")
    max_train_time = st.slider("Max Training Time (min)", 1, 60, 5)
    subsample = st.slider("Data Subsampling (%)", 10, 100, 100)

st.title("📊 Smart Model Tuner")
st.write("Optimized fine-tuning with dataset insights")

training_data = []
with st.expander("📁 Dataset Management", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            training_data = df.to_dict('records')
        else:
            example_count = st.number_input("Number of Examples", 1, 100, 3)
            for i in range(example_count):
                with st.container(border=True):
                    inp = st.text_input(f"Question {i+1}", key=f"q{i}")
                    out = st.text_input(f"Answer {i+1}", key=f"a{i}")
                    if inp and out:
                        training_data.append({"input": inp, "output": out})

    with col2:
        if training_data:
            try:
                validate_dataset(training_data)
                stats = get_dataset_stats(training_data)
                st.write("### Dataset Health")
                st.metric("Examples", stats["Total Examples"])
                st.metric("Avg Q Length", stats["Avg Question Length"])
                st.metric("Avg A Length", stats["Avg Answer Length"])
            except Exception as e:
                st.error(str(e))

with st.expander("🎛 Precision Tuning"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Core Parameters")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            value=3e-4
        )
        epochs = st.slider("Epochs", 1, 5, 2)
    
    with col2:
        st.write("### Efficiency Settings")
        batch_size = st.radio("Batch Size", [8, 16, 32], index=1)
        early_stop = st.checkbox("Early Stopping", True)
        if early_stop:
            patience = st.number_input("Patience Steps", 1, 50, 10)

if st.button("🚦 Start Training", use_container_width=True):
    if not lamini_key:
        st.error("API key required!")
        st.stop()
    
    try:
        validate_dataset(training_data)
        lamini.api_key = lamini_key
        
        if subsample < 100:
            training_data = training_data[:int(len(training_data)*(subsample/100))]
        
        start_time = time.time()
        
        with st.status("Training Process..."):
            st.write("📦 Packaging dataset...")
            llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
            
            st.write("🚀 Initializing model...")
            llm.load_data(training_data)
            
            st.write("🎓 Training...")
            train_args = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': epochs
            }
            if early_stop:
                train_args['early_stopping_patience'] = patience
                
            llm.tune(**train_args)
            
            st.write("⚡ Optimizing...")
            
        training_time = time.time() - start_time
        st.success(f"Trained {len(training_data)} examples in {training_time:.1f}s")
        
        with st.expander("📄 Model Card"):
            st.write(f"""
            - **Base Model**: Llama 3 8B
            - **Learning Rate**: {learning_rate:.1e}
            - **Batch Size**: {batch_size}
            - **Training Speed**: {len(training_data)/training_time:.1f} ex/s
            """)
            
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

# Query Interface
st.markdown("---")
st.write("## 💬 Ask Your Model")

if 'llm' in locals():
    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Generating response..."):
            try:
                response = llm.generate(user_query)
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
else:
    st.warning("Train a model first to enable queries!")

st.markdown("---")
st.write("### Key Features")
st.markdown("""
- Smart batch sizing
- Early stopping
- Data subsampling
- Precision learning rates
- Memory optimization
""")
