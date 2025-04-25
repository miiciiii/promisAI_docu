# Evaluation of Pre-trained NLP Models for Automated Title Generation

---

## [Introduction]

The rise of Natural Language Processing (NLP) in academic and publishing workflows has amplified the importance of automated content summarization and generation. Among these tasks, title generation is critical—it enhances a document's visibility, relevance, and engagement in both digital libraries and academic indexing platforms.

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

### Tested Models

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

## [Key Findings]

Following a systematic evaluation of 10 pre-trained NLP models using a consistent academic abstract prompt, several insights emerged:

### Top-Performing Models

#### 1. `deepseek-ai/DeepSeek-V3-0324`

- **Model Type**: Multilingual instruction-following LLM  
- **Architecture**: DeepSeek Transformer (based on LLaMA)  
- **Strengths**:
  - Produces structured, publication-ready titles
  - Captures abstract-level intent and nuances
  - Effective across multilingual and domain-specific content  
- **Limitations**:
  - May occasionally overgeneralize or reinterpret details  
- **Use Case Fit**: Excellent for academic writing support and policy-aligned research projects  
- **Source**: [Github](https://github.com/deepseek-ai/DeepSeek-V3)

---

#### 2. `google/flan-t5-base`

- **Model Type**: Instruction-tuned sequence-to-sequence  
- **Architecture**: T5 (Text-to-Text Transfer Transformer)  
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

#### 3. `deep-learning-analytics/automatic-title-generation`

- **Model Type**: Task-specific academic title generator  
- **Architecture**: T5-base fine-tuned on academic corpora  
- **Strengths**:
  - High fidelity to research abstracts
  - Generates concise, accurate, domain-specific titles  
- **Limitations**:
  - May underperform in general or creative tasks
  - Outputs tend to be conservative in phrasing  
- **Use Case Fit**: Best suited for journal-style titles and institutional research platforms  
- **Source**: [Hugging Face](https://huggingface.co/deep-learning-analytics/automatic-title-generation)

---

### Highlight Examples

- **Model 1**: *AI in Higher Education: Transforming Teaching, Learning, and Institutional Challenges* — well-rounded and context-aware  
- **Model 8**: *Artificial Intelligence in Higher Education: Benefits and Challenges* — clear and structured  
- **Model 3**: *AI-Powered Teaching and Learning: A Mixed-Method Survey* — aligned with academic conventions  

---

### Other Observations

- **Model 4** (AraT5) produced Arabic output, not suitable for English-centric evaluation.  
- **Models 6, 7, 9** often produced overly extractive or sentence-like outputs rather than true titles.  
- **Model 10** offered partial paraphrases instead of full summarizations.

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

### Integration Plan

1. **Base Model Selection**  
   - Selected: `DeepSeek-V3`, `FLAN-T5`, and `T5-Academic`
   - Basis: Accuracy, domain sensitivity, and instruction-following strength

2. **Dataset Preparation**  
   - Source: PROMIS historical project submissions  
   - Format: JSON or DataFrame with fields:  
     `Abstract`, `Title`, `SDG`, `R&D Priority Area & Program`, `Economic Agenda`  
   - Volume: 500–2,000 labeled entries for initial transfer learning

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
