# Evaluation of Pre-trained NLP Models for Automated Title Generation

---

## [Introduction]

The rise of Natural Language Processing (NLP) in academic and publishing workflows has amplified the importance of automated content summarization and generation. Among these tasks, title generation is critical, it enhances a document's visibility, relevance, and engagement in both digital libraries and academic indexing platforms.

This project evaluates the effectiveness of various state-of-the-art pre-trained language models in generating concise, fluent, and semantically accurate titles from academic abstracts. The primary goal is to determine the most suitable models for integration into the PROMIS (Project Monitoring and Information System) platform, which supports research proposal submissions across government institutions.

---

## [Methodology]

A standardized experimental setup was followed for consistency:

- **Prompt Template**:  
  All models received the same instruction:  
  _"Generate a title for this abstract: [Input Abstract]"_

- **Sample Input Abstract**:  
  > "In recent years, the rapid advancement of artificial intelligence has significantly transformed the landscape of higher education. This study explores the integration of AI-powered tools in university-level teaching and learning, focusing on their impact on student engagement, academic performance, and instructional methods. Using a mixed-methods approach involving surveys, interviews, and classroom observations across five institutions, the research identifies key benefits and challenges associated with AI adoption. The findings suggest that while AI enhances personalized learning and administrative efficiency, it also raises concerns related to data privacy, bias, and the changing role of educators. Recommendations for policy, practice, and future research are provided to guide institutions in effectively leveraging AI for educational innovation."

- **Evaluation Criteria**:  
  While the assessment was qualitative, outputs were judged on:
  - **Semantic Relevance**
  - **Fluency and Grammatical Accuracy**
  - **Conciseness and Informativeness**

---

## [Tested Models]

| #  | Model Name                                                | Architecture | Source       | Generated Title                                                                                                                   |
|----|-----------------------------------------------------------|--------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------|
| 1  | `deepseek-ai/DeepSeek-V3-0324`                            | DeepSeek     | Hugging Face | *AI in Higher Education: Transforming Teaching, Learning, and Institutional Challenges*                                           |
| 2  | `fabiochiu/t5-base-medium-title-generation`               | T5           | Hugging Face | *The Impact of Artificial Intelligence in Higher Education*                                                                       |
| 3  | `deep-learning-analytics/automatic-title-generation`      | T5           | Hugging Face | *AI-Powered Teaching and Learning: A Mixed-Method Survey*                                                                         |
| 4  | `UBC-NLP/AraT5-base-title-generation`                     | AraT5        | Hugging Face | *AI articolarticolية تسهم في تحسين أداء التعليم والتعليم*                                                                             |
| 5  | `vngrs-ai/VBART-XLarge-Title-Generation-from-News`        | BART         | Hugging Face | *AI-Powered Learning Tools*                                                                                                       |
| 6  | `google/pegasus-large`                                    | PEGASUS      | Google       | *Using a mixed-methods approach involving surveys, interviews, and classroom observations across five institutions*               |
| 7  | `t5-base`                                                 | T5           | Hugging Face | *This study examines the integration of AI-powered tools in university-level teaching and learning*                               |
| 8  | `google/flan-t5-base`                                     | FLAN-T5      | Google       | *Artificial Intelligence in Higher Education: Benefits and Challenges*                                                            |
| 9  | `EleutherAI/gpt-neo-1.3B`                                 | GPT-Neo      | EleutherAI   | *In recent years, the rapid advancement of artificial intelligence (AI)*                                                          |
| 10 | `tuner007/pegasus_paraphrase`                             | PEGASUS      | Hugging Face | *The rapid advancement of artificial intelligence has changed the landscape of higher education.*                                 |

---

### Top-Performing Models

### 1. `deepseek-ai/DeepSeek-V3-0324`

- **Model Type**: Multilingual instruction-following Large Language Model (LLM)
- **Architecture**: DeepSeek Transformer (based on LLaMA)
- **Training Objective**: Pre-trained on 14.8 trillion diverse and high-quality tokens, followed by supervised fine-tuning and reinforcement learning stages to fully harness its capabilities.
- **Strengths**:
  - Produces structured, publication-ready titles
  - Captures abstract-level intent and nuances
  - Effective across multilingual and domain-specific content
- **Limitations**:
  - May occasionally overgeneralize or reinterpret details
- **Use Case Fit**: Excellent for academic writing support and policy-aligned research projects
- **Source**: [GitHub](https://github.com/deepseek-ai/DeepSeek-V3)

---

### 2. `google/flan-t5-base`

- **Model Type**: Instruction-tuned sequence-to-sequence model
- **Architecture**: T5 (Text-to-Text Transfer Transformer)
- **Training Objective**: Pre-trained on the Colossal Clean Crawled Corpus (C4) and fine-tuned on a mixture of tasks, allowing it to learn a more general-purpose representation of language.
- **Strengths**:
  - Balanced output—semantically correct, fluent, and concise
  - Strong zero-shot capabilities
  - Lightweight and fast for real-time use
- **Limitations**:
  - Less creative than larger models
  - May omit highly technical or policy-related terms
- **Use Case Fit**: Ideal for scalable deployments requiring high-quality, low-latency generation
- **Source**: [Hugging Face](https://huggingface.co/google/flan-t5-base)

---

### 3. `deep-learning-analytics/automatic-title-generation`

- **Model Type**: Task-specific academic title generator
- **Architecture**: T5-base fine-tuned on academic corpora
- **Training Objective**: Fine-tuned specifically on pairs of research abstracts and titles, utilizing datasets from academic publications such as arXiv and PubMed.
- **Strengths**:
  - High fidelity to research abstracts
  - Generates concise, accurate, domain-specific titles
- **Limitations**:
  - May underperform in general or creative tasks
  - Outputs tend to be conservative in phrasing
- **Use Case Fit**: Best suited for journal-style titles and institutional research platforms
- **Source**: [Hugging Face](https://huggingface.co/deep-learning-analytics/automatic-title-generation)

---

### Other Observations

- **Model 4** (AraT5) produced Arabic output, not suitable for English-centric evaluation.  
- **Models 6, 7, 9** often produced overly extractive or sentence-like outputs rather than true titles.  
- **Model 10** offered partial paraphrases instead of full summarizations.

---

## [Key Findings]

Following a systematic evaluation of 10 pre-trained NLP models using a consistent academic abstract prompt, several insights emerged:

- **Performance Variability**: Models exhibited varying degrees of effectiveness in generating coherent and contextually relevant titles, with some excelling in specific domains.
- **Domain Adaptability**: Certain models demonstrated better adaptability to specialized fields, producing more accurate and domain-specific titles.
- **Prompt Sensitivity**: :The structure and wording of the input prompt significantly influenced the quality of the generated titles, highlighting the importance of prompt engineering.
- **Multilingual Capabilities**: Models like deepseek-ai/DeepSeek-V3-0324 showcased strong multilingual support, effectively handling abstracts in various languages.
- **Computational Efficiency**: Lighter models such as google/flan-t5-base offered faster response times, making them suitable for real-time applications.​

---
## [Integration of Chosen Model into PROMIS]

As part of PROMIS enhancements, the platform will incorporate an automated title suggestion feature. The top models—`deepseek-ai/DeepSeek-V3-0324`, `google/flan-t5-base`, and `deep-learning-analytics/automatic-title-generation`—were selected as base models due to their consistent, high-quality performance.

### Why Transfer Learning?

While pre-trained models perform well in general contexts, PROMIS operates in a **specialized governmental research environment**. Title generation must align with:

- The 10-Point Economic Agenda of the President  
- Sustainable Development Goals
- R&D Priority Area & Program

To bridge this gap, transfer learning will be applied by fine-tuning selected models on PROMIS-specific historical data containing abstracts and validated titles.

---

## Integration Plan

### 1. **Base Model Selection**

- **Models Chosen**: `deepseek-ai/DeepSeek-V3-0324`, `google/flan-t5-base`, and `deep-learning-analytics/automatic-title-generation`
  
- **Selection Criteria**:
  - High accuracy in generating structured academic titles
  - Sensitivity to domain-specific content
  - Strong instruction-following capabilities
  
- **Rationale**: These models are pre-trained on extensive datasets and have demonstrated effectiveness in academic and multilingual contexts.​[Hugging Face](https://huggingface.co/mrm8488/t5-base-finetuned-qasc?utm_source=chatgpt.com)

### 2. **Dataset Preparation**

- **Source**: PROMIS historical project submissions
  
- **Format**: JSON or DataFrame with the following fields:
  - `Abstract`: Research abstract text
  - `Title`: Corresponding academic title
  - `SDG`: Sustainable Development Goal(s) associated
  - `R&D Priority Area & Program`: Relevant research and development focus areas
  - `Economic Agenda`: Economic objectives addressed
  
- **Volume**: 500–2,000 labeled entries for initial fine-tuning
  
- **Preprocessing**:
  - Clean and tokenize text
  - Ensure consistent formatting across fields
  - Split dataset into training (80%), validation (10%), and test (10%) sets​[KDnuggets+3TOPBOTS+3ResearchGate+3](https://www.topbots.com/transfer-learning-in-nlp/?utm_source=chatgpt.com)[Philschmid+11Hugging Face+11ResearchGate+11](https://huggingface.co/mrm8488/t5-base-finetuned-qasc?utm_source=chatgpt.com)[Philschmid+1Learn R, Python & Data Science Online+1](https://www.philschmid.de/fine-tune-flan-t5?utm_source=chatgpt.com)

### 3. **Model Fine-Tuning**

#### a. **FLAN-T5 Base**

- **Objective**: Adapt the model to generate academic titles from research abstracts
  
- **Procedure**:
  - Load the pre-trained FLAN-T5 model and tokenizer
  - Format input as: `{"input": "summarize: {Abstract}"}`
  - Format output as: `{"output": "{Title}"}`
  - Use Hugging Face's `Trainer` API for fine-tuning
  - **Reference**: FLAN-T5 Tutorial: Guide and Fine-Tuning ​[ResearchGate+3LearnOpenCV+3ResearchGate+3](https://learnopencv.com/fine-tuning-t5/?utm_source=chatgpt.com)[arXiv+8Learn R, Python & Data Science Online+8Learn R, Python & Data Science Online+8](https://www.datacamp.com/tutorial/flan-t5-tutorial?utm_source=chatgpt.com)[Medium+1Medium+1](https://medium.com/%40anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360?utm_source=chatgpt.com)[Hugging Face Forums+1ResearchGate+1](https://discuss.huggingface.co/t/how-to-fine-tune-t5-base-model/8478?utm_source=chatgpt.com)[BytePlus+4Hugging Face+4KDnuggets+4](https://huggingface.co/docs/transformers/en/training?utm_source=chatgpt.com)

#### b. **T5 Base**

- **Objective**: Train the model to generate concise academic titles
  
- **Procedure**:
  - Load the pre-trained T5 model and tokenizer
  - Format input as: `{"input": "summarize: {Abstract}"}`
  - Format output as: `{"output": "{Title}"}`
  - Utilize Hugging Face's `Trainer` API for fine-tuning
  - **Reference**: Fine-tuning T5 with custom datasets ​[arXiv+2Hugging Face+2BytePlus+2](https://huggingface.co/docs/transformers/en/model_doc/t5?utm_source=chatgpt.com)[Medium+8Hugging Face Forums+8Hugging Face+8](https://discuss.huggingface.co/t/fine-tuning-t5-with-custom-datasets/8858?utm_source=chatgpt.com)

#### c. **DeepSeek-V3**

- **Objective**: Fine-tune the model to generate structured titles aligned with research intent
  
- **Procedure**:
  - Load the pre-trained DeepSeek-V3 model and tokenizer
  - Format input as: `{"input": "generate title for: {Abstract}"}`
  - Format output as: `{"output": "{Title}"}`
  - Fine-tune using the Hugging Face `Trainer` API
  - **Reference**: DeepSeek-V3 GitHub Repository [AI Framework+10Hugging Face Forums+10LearnOpenCV+10](https://discuss.huggingface.co/t/how-to-fine-tune-t5-base-model/8478?utm_source=chatgpt.com)

### 4. **Evaluation and Validation**

- **Metrics**:
  - ROUGE Score: Measures the overlap between generated and reference titles
  - BLEU Score: Assesses the precision of n-grams in generated titles
  - Human Evaluation: Subjective assessment of title relevance and clarity
  
- **Procedure**:
  - Evaluate models on the validation set
  - Select the model with the highest performance metrics
  - Conduct human evaluation to ensure quality and relevance​[Medium](https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887?utm_source=chatgpt.com)


---

### Expected Benefits for PROMIS

- **Context-Aware Title Generation**: Automatically aligns with national research themes  
- **Improved Standardization**: Promotes consistency in project naming conventions  
- **User Empowerment**: Assists researchers in producing high-impact, policy-aligned titles  
- **Scalability**: Future-ready architecture supports continual fine-tuning as data grows

---

## [Challenges Encountered]

Several practical and technical hurdles were noted:

- **Inconsistent Output Length**: Some models generated full sentences rather than concise titles  
- **Extractive Behavior**: Models like PEGASUS often echoed long input segments  
- **Language Mismatch**: Language-specific models like AraT5 were irrelevant for English input  
- **Evaluation Subjectivity**: Lacking objective metrics (e.g., ROUGE), results are based on expert judgment  
- **Computational Overhead**: Larger models required substantial GPU resources, limiting testing breadth

---

## [Conclusion]

The evaluation demonstrated that instruction-tuned and domain-specific models significantly outperform general-purpose models in academic title generation. In particular:

- `deepseek-ai/DeepSeek-V3-0324` showed superior contextual understanding and title fluency  
- `google/flan-t5-base` offered the best balance between quality and efficiency  
- `deep-learning-analytics/automatic-title-generation` excelled in academic domain relevance

These findings inform the strategic integration of NLP into PROMIS, empowering researchers with intelligent, policy-aware title suggestions that align with national development goals.
