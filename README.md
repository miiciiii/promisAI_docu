# Evaluation of Pre-trained NLP Models for Automated Title Generation
---

## [Introduction]

The increasing adoption of Natural Language Processing (NLP) in academic and publishing workflows has underscored the value of automated content summarization and generation. Among these tasks, automated title generation plays a pivotal role in enhancing document accessibility, discoverability, and engagement. This project aims to assess the performance of various state-of-the-art pre-trained language models in generating accurate and contextually relevant titles from academic abstracts.

## [Methodology]

To evaluate the models' capabilities, a consistent experimental setup was established:

- **Prompt Template**:  
  A uniform prompt was used across all models:  
  _"Generate a title for this abstract: [Input Abstract]"_

- **Input Abstract**:  
  > "In recent years, the rapid advancement of artificial intelligence has significantly transformed the landscape of higher education. This study explores the integration of AI-powered tools in university-level teaching and learning, focusing on their impact on student engagement, academic performance, and instructional methods. Using a mixed-methods approach involving surveys, interviews, and classroom observations across five institutions, the research identifies key benefits and challenges associated with AI adoption. The findings suggest that while AI enhances personalized learning and administrative efficiency, it also raises concerns related to data privacy, bias, and the changing role of educators. Recommendations for policy, practice, and future research are provided to guide institutions in effectively leveraging AI for educational innovation."

- **Evaluation Criteria**:  
  Although this was a qualitative assessment, the models were evaluated based on:
  - Semantic relevance of the generated title
  - Fluency and grammatical correctness
  - Conciseness and informativeness

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

## [Key Findings]

After testing ten pre-trained models on a consistent academic abstract prompt, the following observations were made:

- **Top-Performing Models**:
  - **Model 1** ([`deepseek-ai/DeepSeek-V3-0324`](https://github.com/deepseek-ai/DeepSeek-V3))
  - **Model 8** ([`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base))
  - **Model 3** ([`deep-learning-analytics/automatic-title-generation`](https://huggingface.co/deep-learning-analytics/automatic-title-generation))

  These models consistently produced titles that were:
  - Semantically aligned with the abstract
  - Grammatically fluent and professionally phrased
  - Concise while retaining key information

- **Highlights**:
  - **Model 1**: *AI in Higher Education: Transforming Teaching, Learning, and Institutional Challenges* — comprehensive and academically styled
  - **Model 8**: *Artificial Intelligence in Higher Education: Benefits and Challenges* — well-balanced and informative
  - **Model 3**: *AI-Powered Teaching and Learning: A Mixed-Method Survey* — clear research focus

- **Other Observations**:
  - Some models (e.g., Models 6, 7, 9) generated outputs that were extractive or lacked brevity and structure.
  - **Model 4**, designed for Arabic, produced a region-specific output, making it unsuitable for English-based evaluation.

These findings suggest that instruction-tuned and task-specialized models (e.g., FLAN-T5 and DeepSeek) are most effective for academic title generation in English.

## [Challenges]

During the evaluation, the following challenges were encountered:

- **Inconsistent Output Length and Format**:  
  General-purpose models like GPT-Neo and standard T5 often generated sentences instead of succinct titles.

- **Extractive Behavior**:  
  Models such as PEGASUS frequently pulled long excerpts from the input instead of abstracting the content.

- **Language-Specific Output**:  
  The Arabic model (`AraT5`) produced multilingual titles, which, while accurate for its domain, didn't align with the English-only context.

- **Subjective Evaluation**:  
  Without formal metrics like ROUGE or BLEU, performance evaluation was qualitative and may lack replicability.

- **Resource Constraints**:  
  Large models (e.g., PEGASUS-Large, GPT-Neo 1.3B) demanded significant computational power, limiting scalability in testing.

## [Conclusions]

This evaluation assessed ten pre-trained NLP models for academic title generation from abstracts. Key takeaways include:

- Models fine-tuned specifically for title generation (e.g., `deepseek-ai/DeepSeek-V3-0324`, `deep-learning-analytics/automatic-title-generation`, `fabiochiu/t5-base-medium-title-generation`) consistently outperformed more general-purpose models.
- Instruction-tuned models like `google/flan-t5-base` displayed excellent adaptability and language generation capabilities.
- Extractive models and those without task-specific fine-tuning underperformed in generating concise, structured titles.
