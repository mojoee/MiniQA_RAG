## TODOs

1. Model Selection

Choose a model from Hugging Face's Transformers library for its balance between performance and efficiency. If computational resources allow, consider larger models for better accuracy.

For a balance between performance and efficiency, consider the following models available on Hugging Face's Transformers library:

### 1. **DistilBERT**
- **Model:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- **Description:** DistilBERT is a smaller, faster, and lighter version of BERT that retains 97% of BERT’s language understanding. It's optimized for efficiency, making it a good choice if computational resources are limited.
- **Performance:** High efficiency, slightly lower accuracy compared to larger BERT models.

### 2. **RoBERTa**
- **Model:** [roberta-base](https://huggingface.co/roberta-base)
- **Description:** RoBERTa is an optimized version of BERT with improved training techniques and larger training data. It offers a good balance between performance and computational efficiency.
- **Performance:** High accuracy with moderate resource requirements.

### 3. **ALBERT**
- **Model:** [albert-base-v2](https://huggingface.co/albert-base-v2)
- **Description:** ALBERT is designed to be more parameter-efficient than BERT, achieving comparable performance with fewer parameters.
- **Performance:** Excellent parameter efficiency, with good accuracy and lower resource requirements.

### 4. **BERT**
- **Model:** [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- **Description:** The original BERT model, well-balanced in terms of performance and resource usage. It’s a versatile choice for many natural language processing tasks.
- **Performance:** Very high accuracy with moderate resource requirements.

### 5. **GPT-2**
- **Model:** [gpt2](https://huggingface.co/gpt2)
- **Description:** GPT-2 is a transformer-based model designed for generating coherent and contextually relevant text. It’s efficient in terms of generation tasks.
- **Performance:** High-quality text generation with moderate to high resource requirements.

### 6. **TinyBERT**
- **Model:** [huawei-noah/TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
- **Description:** TinyBERT is a distilled version of BERT designed for better efficiency and faster inference while maintaining a high level of performance.
- **Performance:** High efficiency with good performance for many NLP tasks.

Each of these models strikes a balance between performance and efficiency, with specific strengths depending on your particular use case. For general-purpose NLP tasks, `roberta-base` and `bert-base-uncased` are excellent choices. If efficiency is a primary concern, `distilbert-base-uncased` and `TinyBERT` are more suitable. For text generation tasks, `gpt2` is recommended. 

For our case, we will choose: 



2. Corpus Assembly

Assemble a small corpus of legal documents related to the SFC Code of Conduct. This information can be found on their website; you may need to extract data from the link above to form documents for efficient data retrieval. The dataset should be manageable in size (e.g., 2-4 documents) to simplify preprocessing and allow for quick iterations during the live-coding session.



3. Data Preprocessing

   - Write scripts to clean and preprocess the text data. This includes removing special characters, unnecessary whitespace, and potentially summarizing long documents to ensure they are concise and model-friendly.



4. OpenSearch Integration

Set up an OpenSearch instance to index the preprocessed corpus. Write scripts to ingest the documents into OpenSearch, enabling efficient data retrieval during the question-answering process.